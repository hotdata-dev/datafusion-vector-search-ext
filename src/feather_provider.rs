// feather_provider.rs — Arrow/Feather (Arrow IPC) positional PointLookupProvider.
//
// A drop-in alternative to `SqliteLookupProvider` for the DuckLake vector-search
// payload sidecar. The payload is stored as an Arrow IPC file sorted ascending by
// the (sparse / holey) `rowid` key. Hydration is:
//
//   1. a coarse per-batch first-key index narrows to the batch that may hold a
//      requested key,
//   2. `slice::partition_point` binary-searches that batch's rowid column,
//   3. an exact-match guard (`rowid[pos] == key`) rejects a missing rowid, then
//   4. `take` pulls the matched physical rows out of the payload columns.
//
// Memory parity with SQLite (which pages its B-tree) comes from two pieces:
//
//   * READ: the file is `mmap`'d and the Arrow arrays are decoded ZERO-COPY
//     (`FileDecoder` over a `Buffer` backed by the mapping), so column buffers
//     point into the mapped file. `take` over scattered keys faults in only the
//     touched pages; only the small K-row output is heap-allocated. Resident
//     payload memory is bounded by the working set, not the file size.
//
//   * BUILD: a bounded-memory external (spill) sort. Rows stream UNSORTED to a
//     temp IPC file (O(one batch) payload resident); only an O(N)·8 B key index
//     is kept in memory. At `finish` the keys are argsorted and the rows are
//     gathered in sorted order from the mmap'd temp via `interleave`, in bounded
//     chunks, into the final sorted file. No full-payload buffering, and no
//     assumption about the input scan's row order.
//
// Because the payload is stored verbatim Arrow, there is zero type conversion:
// Decimal / Struct / Map / FixedSizeList / Dictionary / Timestamp(tz) / nested
// List all round-trip losslessly (unlike the SQLite provider, which rejects or
// coerces them).

use std::any::Any;
use std::fmt;
use std::fs::File;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::arrow::buffer::Buffer;
use datafusion::arrow::compute;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Result as DFResult;
use datafusion::datasource::MemTable;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;

use crate::lookup::{PointLookupProvider, extract_keys_as_u64};

/// Rows per output batch when gathering the sorted result at build time. Bounds
/// the per-chunk `interleave` working set.
const BUILD_CHUNK_ROWS: usize = 8192;

// ── mmap + zero-copy IPC decode ─────────────────────────────────────────────────

/// Memory-map `path` and decode every Arrow IPC record batch ZERO-COPY: the
/// returned arrays' buffers point into the mapping (kept alive by the returned
/// [`Buffer`]'s custom allocation owner), so nothing is copied onto the heap.
fn mmap_feather(path: &str) -> DFResult<(SchemaRef, Vec<RecordBatch>, Buffer)> {
    use arrow_ipc::convert::fb_to_schema;
    use arrow_ipc::reader::{FileDecoder, read_footer_length};
    use arrow_ipc::root_as_footer;

    let file = File::open(path)
        .map_err(|e| DataFusionError::Execution(format!("open feather sidecar {path}: {e}")))?;
    // SAFETY: the file is treated as immutable for the provider's lifetime (built
    // once, then read-only). The mapping is owned by the `Buffer` below.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| DataFusionError::Execution(format!("mmap feather sidecar {path}: {e}")))?;
    let len = mmap.len();
    if len < 10 {
        return Err(DataFusionError::Execution(format!(
            "feather sidecar {path} is too small to be a valid Arrow IPC file ({len} bytes)"
        )));
    }
    let ptr = NonNull::new(mmap.as_ptr() as *mut u8).ok_or_else(|| {
        DataFusionError::Execution(format!("feather sidecar {path}: null mmap pointer"))
    })?;
    // SAFETY: `ptr`/`len` describe the mapping, and the `Arc<Mmap>` owner keeps it
    // alive for as long as any slice of this Buffer (or array derived from it) lives.
    let buffer = unsafe { Buffer::from_custom_allocation(ptr, len, Arc::new(mmap)) };

    let trailer_start = len - 10;
    let footer_len = read_footer_length(buffer[trailer_start..].try_into().map_err(|_| {
        DataFusionError::Execution(format!("feather sidecar {path}: bad IPC trailer"))
    })?)
    .map_err(|e| DataFusionError::Execution(format!("feather sidecar {path}: {e}")))?;
    let footer = root_as_footer(&buffer[trailer_start - footer_len..trailer_start])
        .map_err(|e| DataFusionError::Execution(format!("feather sidecar {path}: {e}")))?;

    let schema: SchemaRef = Arc::new(fb_to_schema(footer.schema().ok_or_else(|| {
        DataFusionError::Execution(format!("feather sidecar {path}: footer has no schema"))
    })?));

    let mut decoder = FileDecoder::new(schema.clone(), footer.version());
    for block in footer.dictionaries().iter().flatten() {
        let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
        let data = buffer.slice_with_length(block.offset() as usize, block_len);
        decoder
            .read_dictionary(block, &data)
            .map_err(|e| DataFusionError::Execution(format!("feather sidecar {path}: {e}")))?;
    }

    let mut batches = Vec::new();
    if let Some(record_batches) = footer.recordBatches() {
        for block in record_batches {
            let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
            let data = buffer.slice_with_length(block.offset() as usize, block_len);
            if let Some(batch) = decoder
                .read_record_batch(block, &data)
                .map_err(|e| DataFusionError::Execution(format!("feather sidecar {path}: {e}")))?
            {
                batches.push(batch);
            }
        }
    }
    Ok((schema, batches, buffer))
}

// ── Key helpers (u64 ordering domain — matches `extract_keys_as_u64`) ───────────

/// Read the key at `row` as `u64`. Supports the same integer key types as
/// `extract_keys_as_u64` (Int64/UInt64/Int32/UInt32).
fn key_at(col: &dyn Array, row: usize) -> DFResult<u64> {
    let any = col.as_any();
    if let Some(a) = any.downcast_ref::<Int64Array>() {
        return Ok(a.value(row) as u64);
    }
    if let Some(a) = any.downcast_ref::<UInt64Array>() {
        return Ok(a.value(row));
    }
    if let Some(a) = any.downcast_ref::<Int32Array>() {
        return Ok(a.value(row) as u64);
    }
    if let Some(a) = any.downcast_ref::<UInt32Array>() {
        return Ok(a.value(row) as u64);
    }
    Err(DataFusionError::Execution(format!(
        "FeatherLookupProvider: key column type {:?} is not supported; use Int64/UInt64/Int32/UInt32",
        col.data_type()
    )))
}

/// First index `i` in `col` whose key (compared as `u64`) is `>= target`.
/// Binary search over the sorted key column — touches only the pages it reads.
fn partition_point_u64(col: &dyn Array, target: u64) -> DFResult<usize> {
    let any = col.as_any();
    if let Some(a) = any.downcast_ref::<Int64Array>() {
        return Ok(a.values().partition_point(|&v| (v as u64) < target));
    }
    if let Some(a) = any.downcast_ref::<UInt64Array>() {
        return Ok(a.values().partition_point(|&v| v < target));
    }
    if let Some(a) = any.downcast_ref::<Int32Array>() {
        return Ok(a.values().partition_point(|&v| (v as u64) < target));
    }
    if let Some(a) = any.downcast_ref::<UInt32Array>() {
        return Ok(a.values().partition_point(|&v| (v as u64) < target));
    }
    Err(DataFusionError::Execution(format!(
        "FeatherLookupProvider: key column type {:?} is not supported; use Int64/UInt64/Int32/UInt32",
        col.data_type()
    )))
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// mmap-backed, sorted-by-key Arrow positional [`PointLookupProvider`].
///
/// Holds the payload as zero-copy [`RecordBatch`]es referencing the mmap, in
/// ascending-key order, plus a tiny first-key-per-batch coarse index. Built by
/// [`FeatherSidecarBuilder`] or opened from a `.feather` file with
/// [`open`](Self::open).
pub struct FeatherLookupProvider {
    schema: SchemaRef,
    /// Payload batches, ascending by key and non-overlapping across batches.
    /// Arrays reference the mmap (see `_mmap`).
    batches: Vec<RecordBatch>,
    /// Coarse index: first key of `batches[i]`, ascending. Narrows a lookup to a
    /// single batch before the in-batch binary search.
    batch_first_key: Vec<u64>,
    /// Keeps the memory mapping alive; the batch arrays' buffers point into it.
    _mmap: Buffer,
    key_col: String,
    n_rows: usize,
}

impl fmt::Debug for FeatherLookupProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FeatherLookupProvider(key_col={}, rows={}, batches={}, schema_cols={})",
            self.key_col,
            self.n_rows,
            self.batches.len(),
            self.schema.fields().len()
        )
    }
}

impl FeatherLookupProvider {
    /// Open an existing `.feather` (Arrow IPC file) sidecar via mmap.
    ///
    /// The on-disk schema is self-describing, so the provider's
    /// [`schema`](PointLookupProvider::schema) is taken from the file. Fails if
    /// the stored key column is not strictly ascending — the binary search
    /// depends on it, and the builder always writes it sorted, so a violation
    /// signals a corrupt or foreign file.
    pub fn open(path: &str) -> DFResult<Self> {
        let (schema, batches, mmap) = mmap_feather(path)?;
        Self::from_parts(schema, batches, mmap)
    }

    fn from_parts(schema: SchemaRef, batches: Vec<RecordBatch>, mmap: Buffer) -> DFResult<Self> {
        if schema.fields().is_empty() {
            return Err(DataFusionError::Execution(
                "FeatherLookupProvider: schema has no columns; field 0 must be the key column"
                    .into(),
            ));
        }
        let key_col = schema.field(0).name().clone();

        // Build the coarse index and verify the global ascending+unique invariant
        // the binary search relies on (within each batch and across batches).
        let mut batch_first_key: Vec<u64> = Vec::with_capacity(batches.len());
        let mut n_rows = 0usize;
        let mut prev_last: Option<u64> = None;
        let mut kept: Vec<RecordBatch> = Vec::with_capacity(batches.len());
        for batch in batches {
            let nrows = batch.num_rows();
            if nrows == 0 {
                continue;
            }
            let keycol = batch.column(0).as_ref();
            if keycol.null_count() > 0 {
                return Err(DataFusionError::Execution(
                    "FeatherLookupProvider: key column has a null value; row keys must be non-null"
                        .into(),
                ));
            }
            let first = key_at(keycol, 0)?;
            // Strictly ascending within the batch.
            let mut prev = first;
            for r in 1..nrows {
                let k = key_at(keycol, r)?;
                if k <= prev {
                    return Err(DataFusionError::Execution(format!(
                        "FeatherLookupProvider: key column '{key_col}' is not strictly ascending \
                         (unsorted or duplicate keys); the sidecar is corrupt or was not built by \
                         FeatherSidecarBuilder"
                    )));
                }
                prev = k;
            }
            // Strictly ascending across the batch boundary.
            if prev_last.is_some_and(|pl| first <= pl) {
                return Err(DataFusionError::Execution(format!(
                    "FeatherLookupProvider: key column '{key_col}' is not strictly ascending \
                     across batches; the sidecar is corrupt or was not built by \
                     FeatherSidecarBuilder"
                )));
            }
            prev_last = Some(prev);
            batch_first_key.push(first);
            n_rows += nrows;
            kept.push(batch);
        }

        Ok(Self {
            schema,
            batches: kept,
            batch_first_key,
            _mmap: mmap,
            key_col,
            n_rows,
        })
    }

    pub fn len(&self) -> usize {
        self.n_rows
    }
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
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
        if keys.is_empty() || self.batches.is_empty() {
            return Ok(vec![]);
        }

        // Validate the projection up front and gracefully (like
        // `RecordBatch::project`), rather than panicking on an out-of-range index.
        let nfields = self.schema.fields().len();
        let proj: Vec<usize> = match projection {
            None => (0..nfields).collect(),
            Some(idxs) => {
                if let Some(&bad) = idxs.iter().find(|&&i| i >= nfields) {
                    return Err(DataFusionError::Execution(format!(
                        "FeatherLookupProvider: projection index {bad} out of bounds for schema \
                         with {nfields} columns"
                    )));
                }
                idxs.to_vec()
            }
        };
        let out_schema: SchemaRef = Arc::new(Schema::new(
            proj.iter()
                .map(|&i| self.schema.field(i).clone())
                .collect::<Vec<_>>(),
        ));

        // Sort + dedup the requested keys: sorting yields ascending output
        // (matching SQLite's `ORDER BY rowid`); dedup matches `WHERE key IN (...)`.
        let mut want = keys.to_vec();
        want.sort_unstable();
        want.dedup();

        // Resolve each present key to (batch index, local row), in ascending key
        // order. The coarse index narrows to a batch; partition_point + exact
        // match locate the row within it.
        let mut matches: Vec<(usize, u32)> = Vec::with_capacity(want.len());
        for &k in &want {
            let cb = self.batch_first_key.partition_point(|&fk| fk <= k);
            if cb == 0 {
                continue; // below the smallest stored key
            }
            let b = cb - 1;
            let keycol = self.batches[b].column(0).as_ref();
            let pos = partition_point_u64(keycol, k)?;
            if pos < self.batches[b].num_rows() && key_at(keycol, pos)? == k {
                matches.push((b, pos as u32));
            }
        }
        if matches.is_empty() {
            return Ok(vec![]);
        }

        // `matches` is grouped by batch (ascending k → non-decreasing batch idx).
        // `take` each batch's projected columns once, then concat into one batch.
        let mut out_batches: Vec<RecordBatch> = Vec::new();
        let mut j = 0;
        while j < matches.len() {
            let b = matches[j].0;
            let mut locals: Vec<u32> = Vec::new();
            while j < matches.len() && matches[j].0 == b {
                locals.push(matches[j].1);
                j += 1;
            }
            let idx = UInt32Array::from(locals);
            let cols: Vec<ArrayRef> = proj
                .iter()
                .map(|&c| {
                    compute::take(self.batches[b].column(c).as_ref(), &idx, None)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                })
                .collect::<DFResult<_>>()?;
            out_batches.push(
                RecordBatch::try_new(out_schema.clone(), cols)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
            );
        }

        let combined = compute::concat_batches(&out_schema, &out_batches)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        Ok(vec![combined])
    }
}

// ── TableProvider ─────────────────────────────────────────────────────────────
//
// The payload is already addressable (mmap'd batches), so a full scan is a cheap
// MemTable over them. Lets DataFusion resolve column names when the provider is
// registered as a table.

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
        let mem = MemTable::try_new(self.schema.clone(), vec![self.batches.clone()])?;
        mem.scan(state, projection, &[], None).await
    }
}

// ── Streaming sidecar builder (bounded-memory external sort) ─────────────────────

/// Incremental builder for a [`FeatherLookupProvider`], the Feather analogue of
/// [`SqliteSidecarBuilder`](crate::sqlite_provider::SqliteSidecarBuilder).
///
/// Takes input [`RecordBatch`]es one at a time (e.g. from the DuckLake
/// snapshot-pinned, row-lineage scan), reads each row's key from a designated
/// column, and projects the value columns into the output schema.
///
/// **Bounded memory, order-agnostic.** Feather binary search needs the on-disk
/// key column sorted, but — unlike SQLite's order-independent B-tree — the input
/// scan's row order is not guaranteed (DuckLake `rowid` order from the scan is
/// not reliable). Rather than buffer the whole payload and sort it (O(N)
/// resident) or depend on the scan emitting sorted rowids, this builder spills:
/// [`push_batch`](Self::push_batch) writes rows UNSORTED to a temp IPC file
/// (O(one batch) payload resident), keeping only an O(N)·8 B key index in memory;
/// [`finish`](Self::finish) argsorts the keys and gathers the rows in sorted
/// order from the mmap'd temp via `interleave`, in bounded chunks, into the final
/// sorted file. Duplicate keys are rejected (mirroring SQLite's primary key).
///
/// The first field of `schema` is the key column; fields 1.. are the stored value
/// columns. `key_col_index` / `value_col_indices` index the *input* batches.
pub struct FeatherSidecarBuilder {
    final_path: String,
    schema: SchemaRef,
    key_col_index: usize,
    value_col_indices: Vec<usize>,
    /// Unsorted spill file; rows are appended in push order. Auto-removed on drop.
    temp: tempfile::NamedTempFile,
    writer: arrow_ipc::writer::FileWriter<File>,
    /// Key of each row, in push (spill) order — the only O(N) in-memory state.
    keys: Vec<u64>,
    last_key: Option<u64>,
    input_sorted: bool,
}

impl FeatherSidecarBuilder {
    /// Begin a build targeting `path` (the output `.feather` file). Opens the
    /// temp spill file immediately.
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
        let temp = tempfile::NamedTempFile::new()
            .map_err(|e| DataFusionError::Execution(format!("create feather spill file: {e}")))?;
        let spill = temp
            .reopen()
            .map_err(|e| DataFusionError::Execution(format!("open feather spill file: {e}")))?;
        let writer = arrow_ipc::writer::FileWriter::try_new(spill, &schema)
            .map_err(|e| DataFusionError::Execution(format!("init feather spill writer: {e}")))?;
        Ok(Self {
            final_path: path.to_string(),
            schema,
            key_col_index,
            value_col_indices,
            temp,
            writer,
            keys: Vec::new(),
            last_key: None,
            input_sorted: true,
        })
    }

    /// Project and spill every row of `batch` (unsorted), recording each row's
    /// key. Peak memory is O(one batch) plus the running key index.
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
        for k in extract_keys_as_u64(key_col.as_ref())?.into_iter().flatten() {
            if self.last_key.is_some_and(|prev| k < prev) {
                self.input_sorted = false;
            }
            self.last_key = Some(k);
            self.keys.push(k);
        }

        // Project input columns to output-schema order: key first, then values.
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(self.value_col_indices.len() + 1);
        cols.push(batch.column(self.key_col_index).clone());
        for &ci in &self.value_col_indices {
            cols.push(batch.column(ci).clone());
        }
        let projected = RecordBatch::try_new(self.schema.clone(), cols)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        self.writer
            .write(&projected)
            .map_err(|e| DataFusionError::Execution(format!("spill feather batch: {e}")))?;
        Ok(())
    }

    /// Whether every row arrived in ascending key order (i.e. the spill was
    /// already sorted). Informational.
    pub fn input_was_sorted(&self) -> bool {
        self.input_sorted
    }

    /// Sort by key and write the final sorted `.feather`, then open a provider
    /// over it. Bounded memory: argsort the key index, then gather rows from the
    /// mmap'd spill via `interleave` in `BUILD_CHUNK_ROWS` chunks.
    pub fn finish(mut self) -> DFResult<FeatherLookupProvider> {
        self.writer
            .finish()
            .map_err(|e| DataFusionError::Execution(format!("finalize feather spill: {e}")))?;

        let n = self.keys.len();
        // Stable argsort of the keys (u64 domain), then reject any duplicate —
        // strictly-ascending is the provider's invariant.
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| self.keys[i as usize]);
        for w in order.windows(2) {
            if self.keys[w[0] as usize] >= self.keys[w[1] as usize] {
                return Err(DataFusionError::Execution(
                    "FeatherSidecarBuilder: duplicate row key; keys must be unique".into(),
                ));
            }
        }

        // mmap the unsorted spill; map global row position → (batch, local row).
        let temp_path = self
            .temp
            .path()
            .to_str()
            .ok_or_else(|| DataFusionError::Execution("feather spill path is not UTF-8".into()))?
            .to_string();
        let (_tschema, tbatches, _tmmap) = mmap_feather(&temp_path)?;
        let mut offsets: Vec<usize> = Vec::with_capacity(tbatches.len() + 1);
        let mut acc = 0usize;
        offsets.push(0);
        for b in &tbatches {
            acc += b.num_rows();
            offsets.push(acc);
        }

        let final_file = File::create(&self.final_path).map_err(|e| {
            DataFusionError::Execution(format!("create feather sidecar {}: {e}", self.final_path))
        })?;
        let mut out = arrow_ipc::writer::FileWriter::try_new(final_file, &self.schema)
            .map_err(|e| DataFusionError::Execution(format!("init feather writer: {e}")))?;

        let ncols = self.schema.fields().len();
        let mut start = 0usize;
        while start < n {
            let end = (start + BUILD_CHUNK_ROWS).min(n);
            // (batch, local) pairs for this chunk, in sorted-key order.
            let pairs: Vec<(usize, usize)> = order[start..end]
                .iter()
                .map(|&gp| {
                    let gp = gp as usize;
                    let b = offsets.partition_point(|&o| o <= gp) - 1;
                    (b, gp - offsets[b])
                })
                .collect();
            let cols: Vec<ArrayRef> = (0..ncols)
                .map(|c| {
                    let arrays: Vec<&dyn Array> =
                        tbatches.iter().map(|b| b.column(c).as_ref()).collect();
                    compute::interleave(&arrays, &pairs)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                })
                .collect::<DFResult<_>>()?;
            let batch = RecordBatch::try_new(self.schema.clone(), cols)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            out.write(&batch)
                .map_err(|e| DataFusionError::Execution(format!("write feather batch: {e}")))?;
            start = end;
        }
        out.finish()
            .map_err(|e| DataFusionError::Execution(format!("finalize feather sidecar: {e}")))?;

        tracing::info!(
            "Feather sidecar '{}' built: {} rows, input_already_sorted={}.",
            self.final_path,
            n,
            self.input_sorted,
        );

        // `self.temp` (the spill) is removed when it drops at end of scope.
        FeatherLookupProvider::open(&self.final_path)
    }
}
