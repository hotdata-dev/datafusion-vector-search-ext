// lookup.rs — PointLookupProvider trait + HashKeyProvider implementation.
//
// PointLookupProvider is the primary integration point for storage engines.
// Implement this trait on your native table type to connect it to the
// vector search extension.  HashKeyProvider is the bundled in-memory
// implementation — suitable for tests and small datasets.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Int32Array, Int64Array, RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::arrow::compute;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Trait for efficient row retrieval by primary key.
///
/// Implementors provide O(k) or O(k log N) row lookups — no full-table scan.
/// The `USearchRegistry` requires this trait instead of a bare `TableProvider`
/// to enforce the performance contract at registration time.
///
/// # Contract
///
/// - `schema()` MUST return the Arrow schema of the rows returned by
///   `fetch_by_keys` (when `projection` is `None`).
/// - `fetch_by_keys` MUST return only rows whose key column value is in `keys`.
/// - Keys not found in the table are silently omitted — not an error.
/// - Returned batches must use a schema consistent with `self.schema()`. When
///   `projection` is `None` all columns are returned; when `Some(&[i, j, ...])`
///   only the columns at those indices (of `self.schema()`) are returned.
/// - The `_distance` column is NOT included — the planner attaches it from the
///   ANN search result.
///
/// # Implementing for a production storage engine
///
/// ```rust,ignore
/// #[async_trait]
/// impl PointLookupProvider for MyEngineTable {
///     fn schema(&self) -> SchemaRef { self.schema.clone() }
///
///     async fn fetch_by_keys(
///         &self,
///         keys: &[u64],
///         key_col: &str,
///         projection: Option<&[usize]>,
///     ) -> Result<Vec<RecordBatch>> {
///         // Use your B-tree / LSM / columnar primary key index here.
///         // Only read and return the rows whose key is in `keys`.
///         // Apply `projection` if Some — this avoids reading unused columns.
///         todo!()
///     }
/// }
/// ```
#[async_trait]
pub trait PointLookupProvider: Send + Sync {
    /// Arrow schema of the rows this provider returns (without `_distance`).
    fn schema(&self) -> SchemaRef;

    async fn fetch_by_keys(
        &self,
        keys: &[u64],
        key_col: &str,
        projection: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>>;
}

// ── HashKeyProvider ───────────────────────────────────────────────────────────

/// In-memory [`PointLookupProvider`] backed by a `HashMap<u64 → (batch, row)>`.
///
/// Built from a set of Arrow [`RecordBatch`]es at construction time (one scan).
/// After construction, [`fetch_by_keys`] is O(k) with no further I/O.
///
/// Also implements [`TableProvider`] so it can be registered directly with a
/// DataFusion [`SessionContext`] for SQL column-name resolution.
///
/// ### When to use
///
/// Suitable for tests and for datasets whose row data fits comfortably in
/// memory. For large on-disk tables implement [`PointLookupProvider`] on the
/// storage engine's existing table type.
///
/// [`fetch_by_keys`]: PointLookupProvider::fetch_by_keys
pub struct HashKeyProvider {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    key_index: HashMap<u64, (usize, usize)>,
    key_col: String,
}

impl HashKeyProvider {
    /// Build from a set of Arrow batches, indexed by `key_col`.
    ///
    /// Scans every row once to populate the key index. Returns an error if
    /// `key_col` is not found in `schema` or if the column type is not one of
    /// `UInt64`, `Int64`, `UInt32`, `Int32`.
    pub fn try_new(schema: SchemaRef, batches: Vec<RecordBatch>, key_col: &str) -> Result<Self> {
        let key_col_idx = schema.index_of(key_col).map_err(|_| {
            DataFusionError::Execution(format!(
                "HashKeyProvider: key column '{key_col}' not found in schema"
            ))
        })?;

        let mut key_index: HashMap<u64, (usize, usize)> = HashMap::new();
        for (batch_idx, batch) in batches.iter().enumerate() {
            let keys = extract_keys_as_u64(batch.column(key_col_idx).as_ref())?;
            for (row_idx, k) in keys.into_iter().enumerate() {
                if let Some(key) = k {
                    key_index.insert(key, (batch_idx, row_idx));
                }
            }
        }

        Ok(Self {
            schema,
            batches,
            key_index,
            key_col: key_col.to_string(),
        })
    }

    pub fn len(&self) -> usize {
        self.key_index.len()
    }
    pub fn is_empty(&self) -> bool {
        self.key_index.is_empty()
    }
}

impl fmt::Debug for HashKeyProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HashKeyProvider(key_col={}, rows={})",
            self.key_col,
            self.key_index.len()
        )
    }
}

#[async_trait]
impl PointLookupProvider for HashKeyProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    async fn fetch_by_keys(
        &self,
        keys: &[u64],
        _key_col: &str,
        projection: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }

        let mut batch_masks: Vec<Vec<bool>> = self
            .batches
            .iter()
            .map(|b| vec![false; b.num_rows()])
            .collect();

        for &key in keys {
            if let Some(&(batch_idx, row_idx)) = self.key_index.get(&key) {
                batch_masks[batch_idx][row_idx] = true;
            }
        }

        let mut result: Vec<RecordBatch> = Vec::with_capacity(keys.len());

        for (batch, mask) in self.batches.iter().zip(batch_masks.iter()) {
            if !mask.iter().any(|&b| b) {
                continue;
            }

            let bool_array = BooleanArray::from(mask.clone());
            let filtered_cols: Vec<Arc<dyn Array>> = batch
                .columns()
                .iter()
                .map(|col| {
                    compute::filter(col.as_ref(), &bool_array)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                })
                .collect::<Result<_>>()?;

            let filtered = RecordBatch::try_new(self.schema.clone(), filtered_cols)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

            let out = match projection {
                None => filtered,
                Some(indices) => filtered
                    .project(indices)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
            };

            result.push(out);
        }

        Ok(result)
    }
}

#[async_trait]
impl TableProvider for HashKeyProvider {
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
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mem = MemTable::try_new(self.schema.clone(), vec![self.batches.clone()])?;
        mem.scan(state, projection, &[], None).await
    }
}

// ── Key extraction ────────────────────────────────────────────────────────────

pub(crate) fn extract_keys_as_u64(col: &dyn Array) -> Result<Vec<Option<u64>>> {
    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        return Ok((0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
            .collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i) as u64)
                }
            })
            .collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
        return Ok((0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i) as u64)
                }
            })
            .collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        return Ok((0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i) as u64)
                }
            })
            .collect());
    }
    Err(DataFusionError::Execution(format!(
        "USearch: key column type {:?} is not supported; use UInt64, Int64, UInt32, or Int32",
        col.data_type()
    )))
}
