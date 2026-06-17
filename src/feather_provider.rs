// feather_provider.rs — Arrow/Feather (Arrow IPC) positional PointLookupProvider.
//
// A drop-in alternative to `SqliteLookupProvider` for the DuckLake vector-search
// payload sidecar. Where SQLite stores rows in a B-tree keyed by `rowid` and
// hydrates via `WHERE rowid IN (...)`, this provider stores the payload as a
// single Arrow IPC file sorted ascending by the (sparse / holey) `rowid`, and
// hydrates by:
//
//   1. binary-searching the sorted `rowid` column for each requested key
//      (`slice::partition_point`),
//   2. guarding with an exact-match check (`rowid[pos] == key`) so a missing
//      rowid never aliases its neighbour, then
//   3. `take`-ing the matched physical positions out of the payload columns.
//
// The sorted `rowid` column *is* the index — no separate structure. Because the
// payload is stored verbatim Arrow, hydration is `select(proj).take(positions)`
// with zero type conversion: no `arrow_cell_to_sql` on build, no
// `sql_values_to_arrow` on read, and no rejection/coercion of Decimal / Struct /
// Map / FixedSizeList / Dictionary / Timestamp(tz) / nested List.
//
// Scope: this is the PoC primitive (ticket #702). The whole file is read into
// memory once at open and held resident (the "expand to uncompressed on NVMe
// before querying" model). mmap demand-paging and a coarse per-batch index for
// multi-GB sidecars are documented follow-ups, not implemented here.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::arrow::compute;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Result as DFResult;
use datafusion::datasource::MemTable;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;

use crate::lookup::{PointLookupProvider, extract_keys_as_u64};

// ── Provider ──────────────────────────────────────────────────────────────────

/// In-memory, sorted-by-key Arrow positional [`PointLookupProvider`].
///
/// Holds the full payload as one concatenated [`RecordBatch`] (sorted ascending
/// by the key column, which is field 0 of the schema) plus a contiguous
/// `Vec<u64>` of the key values for binary search. Built by
/// [`FeatherSidecarBuilder`] or opened from an existing `.feather` file with
/// [`open`](Self::open).
pub struct FeatherLookupProvider {
    schema: SchemaRef,
    /// Payload rows, sorted ascending by `keys[i]`. Row `i` of every column
    /// corresponds to `keys[i]`.
    batch: RecordBatch,
    /// Sorted ascending key values, one per row of `batch`. The contiguous
    /// `rowid` index: `partition_point` over this maps a key → physical row.
    keys: Arc<Vec<u64>>,
    key_col: String,
}

impl fmt::Debug for FeatherLookupProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FeatherLookupProvider(key_col={}, rows={}, schema_cols={})",
            self.key_col,
            self.keys.len(),
            self.schema.fields().len()
        )
    }
}

impl FeatherLookupProvider {
    /// Open an existing `.feather` (Arrow IPC file) sidecar.
    ///
    /// Reads every batch, concatenates them into one resident batch, and lifts
    /// the key column (field 0) into a contiguous `Vec<u64>`. The on-disk schema
    /// is self-describing, so the provider's [`schema`](PointLookupProvider::schema)
    /// is taken from the file verbatim. Fails if the stored key column is not
    /// sorted ascending — `fetch_by_keys`' binary search depends on it, and the
    /// builder always writes sorted, so an unsorted file signals corruption.
    pub fn open(path: &str) -> DFResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| DataFusionError::Execution(format!("open feather sidecar {path}: {e}")))?;
        let reader = arrow_ipc::reader::FileReader::try_new(file, None)
            .map_err(|e| DataFusionError::Execution(format!("read feather sidecar {path}: {e}")))?;
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>().map_err(|e| {
            DataFusionError::Execution(format!("decode feather sidecar {path}: {e}"))
        })?;
        Self::from_batches(schema, batches)
    }

    /// Build a provider from already-decoded, sorted-by-key batches. Shared by
    /// [`open`](Self::open) and [`FeatherSidecarBuilder::finish`].
    fn from_batches(schema: SchemaRef, batches: Vec<RecordBatch>) -> DFResult<Self> {
        if schema.fields().is_empty() {
            return Err(DataFusionError::Execution(
                "FeatherLookupProvider: schema has no columns; field 0 must be the key column"
                    .into(),
            ));
        }
        let key_col = schema.field(0).name().clone();

        // One contiguous batch so `take` addresses a global physical position.
        let batch = if batches.is_empty() {
            RecordBatch::new_empty(schema.clone())
        } else {
            compute::concat_batches(&schema, &batches)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
        };

        let keys: Vec<u64> = extract_keys_as_u64(batch.column(0).as_ref())?
            .into_iter()
            .map(|k| {
                k.ok_or_else(|| {
                    DataFusionError::Execution(
                        "FeatherLookupProvider: key column has a null value; \
                         row keys must be non-null"
                            .into(),
                    )
                })
            })
            .collect::<DFResult<_>>()?;

        // Binary search requires ascending keys. The builder guarantees this;
        // verify defensively so a bad file fails loudly at open, not silently
        // at query time with wrong rows.
        if keys.windows(2).any(|w| w[0] > w[1]) {
            return Err(DataFusionError::Execution(format!(
                "FeatherLookupProvider: key column '{key_col}' is not sorted ascending; \
                 the sidecar is corrupt or was not built by FeatherSidecarBuilder"
            )));
        }

        Ok(Self {
            schema,
            batch,
            keys: Arc::new(keys),
            key_col,
        })
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

#[async_trait]
impl PointLookupProvider for FeatherLookupProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    async fn fetch_by_keys(
        &self,
        keys: &[u64],
        _key_col: &str,
        projection: Option<&[usize]>,
    ) -> DFResult<Vec<RecordBatch>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }

        // Sort + dedup the requested keys. Sorting makes the resulting physical
        // positions ascending, so the output is ordered by key — matching
        // SQLite's `ORDER BY rowid`. Dedup matches `WHERE key IN (...)`, which
        // returns one row per distinct key regardless of repeats.
        let mut want = keys.to_vec();
        want.sort_unstable();
        want.dedup();

        // Map each requested key to a physical position via binary search, with
        // an exact-match guard so an absent key is skipped rather than aliased
        // onto its lower-bound neighbour.
        let mut positions: Vec<u64> = Vec::with_capacity(want.len());
        for &k in &want {
            let pos = self.keys.partition_point(|&v| v < k);
            if pos < self.keys.len() && self.keys[pos] == k {
                positions.push(pos as u64);
            }
        }
        if positions.is_empty() {
            return Ok(vec![]);
        }

        // Columns to read, in output order. Projection indexes into the provider
        // schema (0 = key column), mirroring the SQLite provider's contract.
        let col_indices: Vec<usize> = match projection {
            None => (0..self.schema.fields().len()).collect(),
            Some(idxs) => idxs.to_vec(),
        };
        let out_schema: SchemaRef = match projection {
            None => self.schema.clone(),
            Some(idxs) => Arc::new(arrow_schema::Schema::new(
                idxs.iter()
                    .map(|&i| self.schema.field(i).clone())
                    .collect::<Vec<_>>(),
            )),
        };

        let pos_arr = UInt64Array::from(positions);
        let cols: Vec<ArrayRef> = col_indices
            .iter()
            .map(|&i| {
                compute::take(self.batch.column(i).as_ref(), &pos_arr, None)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
            })
            .collect::<DFResult<_>>()?;

        let batch = RecordBatch::try_new(out_schema, cols)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        Ok(vec![batch])
    }
}

// ── TableProvider ─────────────────────────────────────────────────────────────
//
// Mirrors `SqliteLookupProvider` / `HashKeyProvider`: the payload is already
// resident, so a full scan is a cheap MemTable over the single batch. This lets
// DataFusion resolve column names when the provider is registered as a table.

#[async_trait]
impl TableProvider for FeatherLookupProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let mem = MemTable::try_new(self.schema.clone(), vec![vec![self.batch.clone()]])?;
        mem.scan(state, projection, &[], None).await
    }
}

// ── Streaming sidecar builder ───────────────────────────────────────────────

/// Incremental builder for a [`FeatherLookupProvider`], the Feather analogue of
/// [`SqliteSidecarBuilder`](crate::sqlite_provider::SqliteSidecarBuilder).
///
/// Takes input [`RecordBatch`]es one at a time (e.g. from the DuckLake
/// snapshot-pinned, row-lineage scan), reading each row's key from a designated
/// column and projecting the value columns into the output schema. On
/// [`finish`](Self::finish) the buffered rows are sorted ascending by key and
/// written as one Arrow IPC batch.
///
/// **Sort-agnostic by design.** Unlike the SQLite B-tree (which is order
/// independent), Feather binary search requires the on-disk key column sorted.
/// Rather than depend on the DuckLake scan emitting sorted rowids — the ticket's
/// top open risk — this builder sorts at `finish`, so correctness holds for any
/// input order. Whether the input was *already* sorted is tracked and reported
/// via [`input_was_sorted`](Self::input_was_sorted), feeding the decision on
/// whether a bounded-memory merge (instead of the in-memory sort) is needed at
/// production scale.
///
/// **Memory:** the PoC buffers all projected rows before sorting — O(N) resident.
/// A bounded-memory external merge-of-sorted-runs is the documented scale plan;
/// it is not implemented here.
///
/// The first field of `schema` is the key column; fields 1.. are the stored
/// value columns. `key_col_index` / `value_col_indices` index into the *input*
/// batches passed to [`push_batch`](Self::push_batch) (matching
/// [`SqliteSidecarBuilder::begin`](crate::sqlite_provider::SqliteSidecarBuilder::begin)).
pub struct FeatherSidecarBuilder {
    path: String,
    schema: SchemaRef,
    key_col_index: usize,
    value_col_indices: Vec<usize>,
    /// Projected batches conforming to `schema`, accumulated across push_batch.
    buffered: Vec<RecordBatch>,
    /// Largest key seen so far, to detect whether the input stream is already
    /// globally sorted ascending.
    last_key: Option<u64>,
    input_sorted: bool,
}

impl FeatherSidecarBuilder {
    /// Begin a build targeting `path` (the output `.feather` file).
    ///
    /// `schema` is the output schema — field 0 is the key column, fields 1.. are
    /// the stored value columns, verbatim Arrow (no type validation: storing
    /// types SQLite rejects is the point). `key_col_index` and
    /// `value_col_indices` index into the input batches.
    pub fn begin(
        path: &str,
        schema: SchemaRef,
        key_col_index: usize,
        value_col_indices: Vec<usize>,
    ) -> DFResult<Self> {
        if schema.fields().is_empty() {
            return Err(DataFusionError::Execution(
                "FeatherSidecarBuilder: schema has no columns; field 0 must be the key column"
                    .into(),
            ));
        }
        if schema.fields().len() != value_col_indices.len() + 1 {
            return Err(DataFusionError::Execution(format!(
                "FeatherSidecarBuilder: schema has {} fields but expected 1 key column + {} value columns",
                schema.fields().len(),
                value_col_indices.len()
            )));
        }
        Ok(Self {
            path: path.to_string(),
            schema,
            key_col_index,
            value_col_indices,
            buffered: Vec::new(),
            last_key: None,
            input_sorted: true,
        })
    }

    /// Project and buffer every row of `batch`. The key column is read from
    /// `key_col_index`; value columns from `value_col_indices`, in order, into
    /// schema fields 1.. . Type mismatches between the input columns and the
    /// declared schema surface here (via `RecordBatch::try_new`).
    pub fn push_batch(&mut self, batch: &RecordBatch) -> DFResult<()> {
        let ncols = batch.num_columns();
        if self.key_col_index >= ncols {
            return Err(DataFusionError::Execution(format!(
                "FeatherSidecarBuilder: key_col_index {} out of range for batch with {ncols} columns",
                self.key_col_index
            )));
        }
        if let Some(&bad) = self.value_col_indices.iter().find(|&&i| i >= ncols) {
            return Err(DataFusionError::Execution(format!(
                "FeatherSidecarBuilder: value column index {bad} out of range for batch with {ncols} columns"
            )));
        }

        let key_col = batch.column(self.key_col_index);
        if key_col.null_count() > 0 {
            return Err(DataFusionError::Execution(
                "FeatherSidecarBuilder: key column has a null value; row keys must be non-null"
                    .into(),
            ));
        }

        // Track global sortedness while we have the keys in hand.
        let keys = extract_keys_as_u64(key_col.as_ref())?;
        for k in keys.into_iter().flatten() {
            if self.last_key.is_some_and(|prev| k < prev) {
                self.input_sorted = false;
            }
            self.last_key = Some(k);
        }

        // Project input columns into output-schema order: key first, then values.
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(self.value_col_indices.len() + 1);
        cols.push(batch.column(self.key_col_index).clone());
        for &ci in &self.value_col_indices {
            cols.push(batch.column(ci).clone());
        }
        let projected = RecordBatch::try_new(self.schema.clone(), cols)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        self.buffered.push(projected);
        Ok(())
    }

    /// Whether every row pushed so far arrived in ascending key order (i.e. the
    /// in-memory sort at `finish` was a no-op). Informational: feeds the
    /// production decision on whether the DuckLake scan can be relied on to emit
    /// sorted rowids (skipping the sort) or needs a merge step.
    pub fn input_was_sorted(&self) -> bool {
        self.input_sorted
    }

    /// Sort buffered rows ascending by key, write them as a single Arrow IPC
    /// batch to `path`, and open a [`FeatherLookupProvider`] over the result.
    pub fn finish(self) -> DFResult<FeatherLookupProvider> {
        let combined = if self.buffered.is_empty() {
            RecordBatch::new_empty(self.schema.clone())
        } else {
            compute::concat_batches(&self.schema, &self.buffered)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
        };

        // Sort ascending by the key column (field 0). Always performed so the
        // file is sorted regardless of input order; cheap when already sorted.
        let sorted = if combined.num_rows() == 0 {
            combined
        } else {
            let indices = compute::sort_to_indices(combined.column(0).as_ref(), None, None)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            sort_batch(&combined, &indices)?
        };

        write_ipc_file(&self.path, &sorted)?;

        tracing::info!(
            "Feather sidecar '{}' built: {} rows, input_already_sorted={}.",
            self.path,
            sorted.num_rows(),
            self.input_sorted,
        );

        FeatherLookupProvider::from_batches(self.schema, vec![sorted])
    }
}

/// `take` every column of `batch` by `indices`, preserving the schema.
fn sort_batch(batch: &RecordBatch, indices: &UInt32Array) -> DFResult<RecordBatch> {
    let cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| {
            compute::take(c.as_ref(), indices, None)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
        })
        .collect::<DFResult<_>>()?;
    RecordBatch::try_new(batch.schema(), cols)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// Write a single batch as an Arrow IPC *file* (Feather v2). Uncompressed: the
/// fast path needs on-disk bytes == in-memory layout (zero-copy take), and IPC
/// whole-buffer compression would force a full-column decode per scattered row.
fn write_ipc_file(path: &str, batch: &RecordBatch) -> DFResult<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| DataFusionError::Execution(format!("create feather sidecar {path}: {e}")))?;
    let mut writer = arrow_ipc::writer::FileWriter::try_new(file, &batch.schema())
        .map_err(|e| DataFusionError::Execution(format!("init feather writer {path}: {e}")))?;
    if batch.num_rows() > 0 {
        writer
            .write(batch)
            .map_err(|e| DataFusionError::Execution(format!("write feather batch {path}: {e}")))?;
    }
    writer
        .finish()
        .map_err(|e| DataFusionError::Execution(format!("finalize feather sidecar {path}: {e}")))?;
    Ok(())
}
