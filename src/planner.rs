// planner.rs — USearchQueryPlanner + USearchExecPlanner + USearchExec.
//
// The extension planner translates USearchNode (logical) into USearchExec
// (physical).  Two execution paths depending on whether the node carries
// absorbed WHERE-clause filters:
//
// ── No filters (original path) ───────────────────────────────────────────────
//   1. index.search() → (keys, distances)
//   2. provider.fetch_by_keys() → exactly those k rows, O(k)
//   3. Attach _distance column.
//
// ── With filters (adaptive path) ─────────────────────────────────────────────
//   1. Compile filter Exprs to PhysicalExprs.
//   2. Full scan of the provider; evaluate filters per batch.
//      Collect: valid_keys (HashSet<u64>) and per-valid-row (key, distance).
//   3. selectivity = valid_keys.len() / index.size()
//   4a. selectivity > threshold → filtered_search(query, k, |key| valid_keys.contains(key))
//       Result: O(k) exact rows; correctness guaranteed by in-graph predicate.
//   4b. selectivity ≤ threshold → skip HNSW; use pre-computed (key, distance)
//       pairs, heap-select top-k, then fetch_by_keys.
//       Cost: O(|valid| × d) ≪ O((k/sel) × M × d) at low selectivity.
//
// All I/O is deferred to USearchExec::execute() — plan_extension is purely
// structural (validate registry entry + compile PhysicalExprs).
//
// The Sort node is kept in the logical plan so DataFusion handles ordering
// by _distance / dist alias.

use std::any::Any;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
    RecordBatch,
};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr, create_physical_expr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};

use futures::StreamExt;
use usearch::ScalarKind;

use tracing::Instrument;

use crate::lookup::extract_keys_as_u64;
use crate::node::{DistanceType, USearchNode};
use crate::registry::USearchRegistry;

// ── QueryPlanner wrapper ──────────────────────────────────────────────────────

pub struct USearchQueryPlanner {
    inner: DefaultPhysicalPlanner,
}

impl fmt::Debug for USearchQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "USearchQueryPlanner")
    }
}

impl USearchQueryPlanner {
    pub fn new(registry: Arc<USearchRegistry>) -> Self {
        let inner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            USearchExecPlanner::new(registry),
        )]);
        Self { inner }
    }
}

#[async_trait]
impl QueryPlanner for USearchQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

// ── Extension planner ─────────────────────────────────────────────────────────

pub struct USearchExecPlanner {
    registry: Arc<USearchRegistry>,
}

impl USearchExecPlanner {
    pub fn new(registry: Arc<USearchRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl ExtensionPlanner for USearchExecPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        _physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let node = match node.as_any().downcast_ref::<USearchNode>() {
            Some(n) => n,
            None => return Ok(None),
        };

        // Cheap validation: RwLock read + HashMap lookup — no I/O.
        let registered = match self.registry.get(&node.table_name) {
            Some(r) => r,
            None => {
                return Err(DataFusionError::Execution(format!(
                    "USearchExecPlanner: table '{}' not in registry",
                    node.table_name
                )));
            }
        };

        // Compile filter Exprs → PhysicalExprs (synchronous, no I/O).
        let exec_props = session_state.execution_props();
        let physical_filters: Vec<Arc<dyn PhysicalExpr>> = node
            .filters
            .iter()
            .map(|f| create_physical_expr(f, &node.schema, exec_props))
            .collect::<Result<_>>()?;

        // For the filtered path, pre-plan two provider scans:
        // 1. Pre-scan (scalar + _key only, no vector col) — cheap selectivity estimation.
        // 2. Full scan (all cols including vector) — only executed at runtime if low-sel.
        // Both scans receive the node's WHERE filters for Parquet predicate pushdown.
        let (provider_scan, full_scan) = if !node.filters.is_empty() {
            let scan_schema = registered.scan_provider.schema();
            let vec_col_idx = scan_schema.index_of(&node.vector_col).ok();

            // Pre-scan projection: all columns except vector.
            let scalar_projection: Vec<usize> = (0..scan_schema.fields().len())
                .filter(|&i| Some(i) != vec_col_idx)
                .collect();

            let pre_scan = registered
                .scan_provider
                .scan(session_state, Some(&scalar_projection), &node.filters, None)
                .await?;

            // Full scan: all columns (vector included for distance computation).
            // Only created if the vector column exists in the scan provider.
            let full = if vec_col_idx.is_some() {
                Some(
                    registered
                        .scan_provider
                        .scan(session_state, None, &node.filters, None)
                        .await?,
                )
            } else {
                None
            };

            (Some(pre_scan), full)
        } else {
            (None, None)
        };

        Ok(Some(Arc::new(USearchExec::new(SearchParams {
            table_name: node.table_name.clone(),
            registry: self.registry.clone(),
            query_vec: node.query_vec_f64(),
            k: node.k,
            distance_type: node.distance_type.clone(),
            physical_filters,
            schema: registered.schema.clone(),
            key_col: registered.key_col.clone(),
            scalar_kind: registered.scalar_kind,
            vector_col: node.vector_col.clone(),
            brute_force_threshold: registered.config.brute_force_selectivity_threshold,
            provider_scan,
            full_scan,
        }))))
    }
}

// ── Search parameters ─────────────────────────────────────────────────────────

/// All parameters needed to run a USearch query, cloned cheaply into execute().
#[derive(Debug, Clone)]
struct SearchParams {
    table_name: String,
    registry: Arc<USearchRegistry>,
    query_vec: Vec<f64>,
    k: usize,
    distance_type: DistanceType,
    physical_filters: Vec<Arc<dyn PhysicalExpr>>,
    schema: SchemaRef,
    key_col: String,
    scalar_kind: ScalarKind,
    vector_col: String,
    brute_force_threshold: f64,
    /// Pre-planned provider scan for the filtered path (scalar + _key only, no vector col).
    /// Used for selectivity estimation. None for the unfiltered path.
    provider_scan: Option<Arc<dyn ExecutionPlan>>,
    /// Pre-planned full scan (all cols including vector) for the low-selectivity
    /// Parquet-native path. None when unfiltered or when no vector column exists.
    full_scan: Option<Arc<dyn ExecutionPlan>>,
}

// ── Physical execution node ───────────────────────────────────────────────────

/// Leaf execution plan that defers all I/O to execute() time.
#[derive(Debug)]
pub struct USearchExec {
    params: SearchParams,
    properties: PlanProperties,
}

impl USearchExec {
    fn new(params: SearchParams) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(params.schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self { params, properties }
    }
}

impl DisplayAs for USearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "USearchExec: table={}, k={}, filters={}",
            self.params.table_name,
            self.params.k,
            self.params.physical_filters.len()
        )
    }
}

impl ExecutionPlan for USearchExec {
    fn name(&self) -> &str {
        "USearchExec"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "USearchExec is a leaf node and takes no children".to_string(),
            ))
        }
    }

    fn execute(
        &self,
        _partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let params = self.params.clone();
        let stream = futures::stream::once(async move { usearch_execute(params, context).await })
            .flat_map(|result| match result {
                Ok(batches) => futures::stream::iter(batches.into_iter().map(Ok)).left_stream(),
                Err(e) => futures::stream::once(async move { Err(e) }).right_stream(),
            });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.params.schema.clone(),
            stream,
        )))
    }
}

// ── Main async execution fn ───────────────────────────────────────────────────

#[tracing::instrument(
    name = "usearch_execute",
    skip_all,
    fields(
        usearch.table = %params.table_name,
        usearch.k = params.k,
        usearch.filter_count = params.physical_filters.len(),
    )
)]
async fn usearch_execute(
    params: SearchParams,
    task_ctx: Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    // Re-fetch at execute time so cache eviction between plan and execute is handled correctly.
    let registered = params.registry.get(&params.table_name).ok_or_else(|| {
        DataFusionError::Execution(format!(
            "USearchExec: table '{}' not in registry at execute time",
            params.table_name
        ))
    })?;

    if params.physical_filters.is_empty() {
        // ── Unfiltered path ───────────────────────────────────────────────
        let matches = usearch_search(
            &registered.index,
            &params.query_vec,
            params.k,
            params.scalar_kind,
        )?;

        if matches.keys.is_empty() {
            return Ok(vec![]);
        }

        let key_to_dist: HashMap<u64, f32> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .map(|(&k, &d)| (k, d))
            .collect();

        let data_batches = registered
            .lookup_provider
            .fetch_by_keys(&matches.keys, &params.key_col, None)
            .await?;

        let key_col_idx = provider_key_col_idx(&registered)?;
        attach_distances(data_batches, key_col_idx, &key_to_dist, &params.schema)
    } else {
        // ── Adaptive filtered path ────────────────────────────────────────
        let scan = params.provider_scan.clone().ok_or_else(|| {
            DataFusionError::Internal(
                "USearchExec: filtered path has no pre-planned provider scan".into(),
            )
        })?;
        adaptive_filtered_execute(&params, &registered, scan, task_ctx).await
    }
}

// ── Adaptive filtered execution ───────────────────────────────────────────────

#[tracing::instrument(
    name = "usearch_adaptive_filter",
    skip_all,
    fields(
        usearch.table = %params.table_name,
        usearch.k = params.k,
        usearch.filter_count = params.physical_filters.len(),
        usearch.valid_rows = tracing::field::Empty,
        usearch.total_rows = tracing::field::Empty,
        usearch.selectivity = tracing::field::Empty,
        usearch.path = tracing::field::Empty,
        usearch.result_count = tracing::field::Empty,
    )
)]
async fn adaptive_filtered_execute(
    params: &SearchParams,
    registered: &crate::registry::RegisteredTable,
    scan_plan: Arc<dyn ExecutionPlan>,
    task_ctx: Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    // Key column index in the pre-scan output schema (projected, no vector col).
    let pre_scan_schema = scan_plan.schema();
    let scan_key_col_idx = pre_scan_schema.index_of(&registered.key_col).map_err(|_| {
        DataFusionError::Execution(format!(
            "USearchExec: key column '{}' not found in pre-scan schema",
            registered.key_col
        ))
    })?;
    // Key column index in lookup_provider schema — used by attach_distances (high-sel path).
    let lookup_key_col_idx = provider_key_col_idx(registered)?;

    // ── Phase 1: Pre-scan (scalar + _key only) for selectivity estimation ────
    let mut stream = scan_plan.execute(0, task_ctx.clone())?;
    let mut valid_keys: HashSet<u64> = HashSet::new();

    let scan_span = tracing::info_span!("usearch_pre_scan", usearch.table = %params.table_name);
    async {
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let mask = evaluate_filters(&params.physical_filters, &batch)?;
            let keys = extract_keys_as_u64(batch.column(scan_key_col_idx).as_ref())?;

            for row_idx in 0..batch.num_rows() {
                if !mask.is_null(row_idx)
                    && mask.value(row_idx)
                    && let Some(Some(key)) = keys.get(row_idx)
                {
                    valid_keys.insert(*key);
                }
            }
        }
        Ok::<_, datafusion::error::DataFusionError>(())
    }
    .instrument(scan_span)
    .await?;

    // No rows pass the filter — return empty.
    if valid_keys.is_empty() {
        tracing::Span::current().record("usearch.valid_rows", 0usize);
        tracing::Span::current().record("usearch.result_count", 0usize);
        return Ok(vec![]);
    }

    let total = registered.index.size();
    let selectivity = valid_keys.len() as f64 / total.max(1) as f64;
    let threshold = params.brute_force_threshold;
    let has_full_scan = params.full_scan.is_some();

    let path = if selectivity <= threshold && has_full_scan {
        "parquet-native"
    } else {
        "filtered_search"
    };
    tracing::Span::current().record("usearch.valid_rows", valid_keys.len());
    tracing::Span::current().record("usearch.total_rows", total);
    tracing::Span::current().record("usearch.selectivity", selectivity);
    tracing::Span::current().record("usearch.path", path);

    if selectivity <= threshold && has_full_scan {
        // ── Low-selectivity: Parquet-native path (no USearch, no SQLite) ──
        // Second scan reads all columns including vector. Compute distances
        // inline, maintain top-k heap, build result directly from scan data.
        let full_scan = params.full_scan.clone().unwrap();
        parquet_native_execute(params, registered, full_scan, task_ctx).await
    } else {
        // ── High-selectivity: HNSW filtered_search + SQLite fetch ─────────
        let matches = tracing::info_span!(
            "usearch_hnsw_filtered_search",
            usearch.table = %params.table_name
        )
        .in_scope(|| {
            usearch_filtered_search(
                &registered.index,
                &params.query_vec,
                params.k,
                registered.scalar_kind,
                |key| valid_keys.contains(&key),
            )
        })?;

        if matches.keys.is_empty() {
            tracing::Span::current().record("usearch.result_count", 0usize);
            return Ok(vec![]);
        }

        let key_to_dist: HashMap<u64, f32> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .map(|(&k, &d)| (k, d))
            .collect();

        let data_batches = registered
            .lookup_provider
            .fetch_by_keys(&matches.keys, &params.key_col, None)
            .await?;

        let result_batches = attach_distances(
            data_batches,
            lookup_key_col_idx,
            &key_to_dist,
            &params.schema,
        )?;

        tracing::Span::current().record(
            "usearch.result_count",
            result_batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        );
        Ok(result_batches)
    }
}

// ── Parquet-native low-selectivity execution ──────────────────────────────────

/// Execute the low-selectivity path entirely from Parquet — no USearch, no SQLite.
///
/// Streams the full scan (all columns including vector), evaluates WHERE filters,
/// computes distances for passing rows, maintains a top-k heap, and builds the
/// result directly from scan data.
#[tracing::instrument(
    name = "usearch_parquet_native",
    skip_all,
    fields(usearch.table = %params.table_name)
)]
async fn parquet_native_execute(
    params: &SearchParams,
    registered: &crate::registry::RegisteredTable,
    full_scan: Arc<dyn ExecutionPlan>,
    task_ctx: Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    let full_schema = full_scan.schema();
    let vec_col_idx = full_schema.index_of(&params.vector_col).map_err(|_| {
        DataFusionError::Execution(format!(
            "USearchExec: vector column '{}' not found in full scan schema",
            params.vector_col
        ))
    })?;

    // Compute output projection: full scan columns minus the vector column.
    // This matches the lookup_provider schema (same column order).
    let output_col_indices: Vec<usize> = (0..full_schema.fields().len())
        .filter(|&i| i != vec_col_idx)
        .collect();

    // Top-k heap: stores (distance, projected_row_slice).
    // At low selectivity (<=5%), the number of passing rows is small.
    let mut heap: BinaryHeap<ScoredRow> = BinaryHeap::with_capacity(params.k + 1);

    let mut stream = full_scan.execute(0, task_ctx)?;
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        let mask = evaluate_filters(&params.physical_filters, &batch)?;

        for row_idx in 0..batch.num_rows() {
            if mask.is_null(row_idx) || !mask.value(row_idx) {
                continue;
            }

            let dist = match compute_distance_for_row(
                &batch,
                vec_col_idx,
                row_idx,
                &params.query_vec,
                registered.scalar_kind,
                &params.distance_type,
            ) {
                Ok(d) => d,
                Err(_) => continue, // skip null vectors
            };

            // Project the row to output columns (drop vector col), zero-copy slice.
            let row_cols: Vec<Arc<dyn Array>> = output_col_indices
                .iter()
                .map(|&i| batch.column(i).slice(row_idx, 1))
                .collect();

            heap.push(ScoredRow {
                distance: dist,
                row: row_cols,
            });
            if heap.len() > params.k {
                heap.pop(); // evict farthest
            }
        }
    }

    if heap.is_empty() {
        tracing::Span::current().record("usearch.result_count", 0usize);
        return Ok(vec![]);
    }

    // Build result: sort by distance ascending, concat into batches.
    let mut entries: Vec<ScoredRow> = heap.into_vec();
    entries.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = entries.len();
    let mut result_cols: Vec<Vec<Arc<dyn Array>>> =
        vec![Vec::with_capacity(n); output_col_indices.len()];
    let mut distances: Vec<f32> = Vec::with_capacity(n);

    for entry in &entries {
        for (col_idx, col_slice) in entry.row.iter().enumerate() {
            result_cols[col_idx].push(col_slice.clone());
        }
        distances.push(entry.distance);
    }

    // Concatenate per-column arrays.
    let concat_cols: Vec<Arc<dyn Array>> = result_cols
        .into_iter()
        .map(|slices| {
            let refs: Vec<&dyn Array> = slices.iter().map(|a| a.as_ref()).collect();
            datafusion::arrow::compute::concat(&refs)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
        })
        .collect::<Result<_>>()?;

    // Append _distance column.
    let mut all_cols = concat_cols;
    all_cols.push(Arc::new(Float32Array::from(distances)));

    let result_batch = RecordBatch::try_new(params.schema.clone(), all_cols)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

    tracing::Span::current().record("usearch.result_count", result_batch.num_rows());
    Ok(vec![result_batch])
}

/// A row with its computed distance, for the top-k heap.
/// The max-heap evicts the *farthest* row when it exceeds k.
struct ScoredRow {
    distance: f32,
    /// Column arrays (one per output column), each a single-row slice.
    row: Vec<Arc<dyn Array>>,
}

impl PartialEq for ScoredRow {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for ScoredRow {}
impl PartialOrd for ScoredRow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredRow {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: largest distance at top → gets evicted first.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Less)
    }
}

// ── USearch dispatch helpers ──────────────────────────────────────────────────

/// Call `index.search` with the native scalar type appropriate for the column.
/// Converts the usearch error into a `DataFusionError::Execution`.
fn usearch_search(
    index: &usearch::Index,
    query_f64: &[f64],
    k: usize,
    scalar_kind: ScalarKind,
) -> Result<usearch::ffi::Matches> {
    match scalar_kind {
        ScalarKind::F64 => index
            .search(query_f64, k)
            .map_err(|e| DataFusionError::Execution(format!("USearch search error: {e}"))),
        _ => {
            let q: Vec<f32> = query_f64.iter().map(|&v| v as f32).collect();
            index
                .search(&q, k)
                .map_err(|e| DataFusionError::Execution(format!("USearch search error: {e}")))
        }
    }
}

/// Call `index.filtered_search` with the native scalar type appropriate for the column.
/// Converts the usearch error into a `DataFusionError::Execution`.
fn usearch_filtered_search<F>(
    index: &usearch::Index,
    query_f64: &[f64],
    k: usize,
    scalar_kind: ScalarKind,
    predicate: F,
) -> Result<usearch::ffi::Matches>
where
    F: Fn(u64) -> bool,
{
    match scalar_kind {
        ScalarKind::F64 => index
            .filtered_search(query_f64, k, predicate)
            .map_err(|e| DataFusionError::Execution(format!("USearch filtered_search: {e}"))),
        _ => {
            let q: Vec<f32> = query_f64.iter().map(|&v| v as f32).collect();
            index
                .filtered_search(&q, k, predicate)
                .map_err(|e| DataFusionError::Execution(format!("USearch filtered_search: {e}")))
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// AND all physical filter expressions against a batch.
/// Returns a BooleanArray (one value per row, true = passes all filters).
fn evaluate_filters(
    filters: &[Arc<dyn PhysicalExpr>],
    batch: &RecordBatch,
) -> Result<BooleanArray> {
    use datafusion::arrow::compute;

    if filters.is_empty() {
        return Ok(BooleanArray::from(vec![true; batch.num_rows()]));
    }

    let mut combined: Option<BooleanArray> = None;
    for filter in filters {
        let col_val = filter.evaluate(batch)?;
        let arr = col_val.into_array(batch.num_rows())?;
        let bool_arr = arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("filter expression did not return BooleanArray".into())
            })?
            .clone();

        combined = Some(match combined {
            None => bool_arr,
            Some(prev) => compute::and(&prev, &bool_arr)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
        });
    }
    Ok(combined.unwrap())
}

/// Extract the distance from a single row of a vector column.
///
/// Handles all combinations of outer array type (FixedSizeList / List / LargeList)
/// and inner element type (Float32 / Float64).  The distance is always returned as
/// `f32` — matching the `_distance` column type — regardless of the column's native
/// precision.  Query is accepted as `f64` and cast to the column's native type.
fn compute_distance_for_row(
    batch: &RecordBatch,
    vec_col_idx: usize,
    row_idx: usize,
    query_f64: &[f64],
    scalar_kind: ScalarKind,
    dist_type: &DistanceType,
) -> Result<f32> {
    let col = batch.column(vec_col_idx);

    if col.is_null(row_idx) {
        return Err(DataFusionError::Execution(
            "null vector in brute-force distance computation".into(),
        ));
    }

    // Extract the row's inner array, regardless of outer type.
    let row_arr: Arc<dyn Array> =
        if let Some(fsl) = col.as_any().downcast_ref::<FixedSizeListArray>() {
            fsl.value(row_idx)
        } else if let Some(la) = col.as_any().downcast_ref::<ListArray>() {
            la.value(row_idx)
        } else if let Some(la) = col.as_any().downcast_ref::<LargeListArray>() {
            la.value(row_idx)
        } else {
            return Err(DataFusionError::Execution(format!(
                "vector column type not supported in brute-force path (got {:?})",
                col.data_type()
            )));
        };

    // Dispatch distance computation by the column's native element type.
    match scalar_kind {
        ScalarKind::F64 => {
            let f64_arr = row_arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    DataFusionError::Execution("F64 column: inner array is not Float64Array".into())
                })?;
            let v = f64_arr.values();
            let query = query_f64;
            let dist = match dist_type {
                DistanceType::L2 => v
                    .iter()
                    .zip(query)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>(),
                DistanceType::Cosine => {
                    let dot: f64 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                    let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let norm_q: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let denom = norm_v * norm_q;
                    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
                }
                DistanceType::NegativeDot => -v.iter().zip(query).map(|(a, b)| a * b).sum::<f64>(),
            };
            Ok(dist as f32)
        }
        _ => {
            // F32 (and any other kind): extract as f32, cast query to f32.
            let f32_arr = row_arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "F32 column: inner array is not Float32Array (got {:?})",
                        row_arr.data_type()
                    ))
                })?;
            let v = f32_arr.values();
            let query: Vec<f32> = query_f64.iter().map(|&x| x as f32).collect();
            let dist = match dist_type {
                // L2sq — matches USearch MetricKind::L2sq (no sqrt).
                DistanceType::L2 => v
                    .iter()
                    .zip(&query)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>(),
                // Cosine distance = 1 - cosine_similarity.
                DistanceType::Cosine => {
                    let dot: f32 = v.iter().zip(&query).map(|(a, b)| a * b).sum();
                    let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let denom = norm_v * norm_q;
                    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
                }
                // Negative inner product — matches USearch MetricKind::IP.
                DistanceType::NegativeDot => -v.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>(),
            };
            Ok(dist)
        }
    }
}

/// Index of the key column in the lookup provider schema.
fn provider_key_col_idx(registered: &crate::registry::RegisteredTable) -> Result<usize> {
    registered
        .lookup_provider
        .schema()
        .index_of(&registered.key_col)
        .map_err(|_| {
            DataFusionError::Execution(format!(
                "USearchExecPlanner: key column '{}' not found in lookup provider schema",
                registered.key_col
            ))
        })
}

// ── Distance attachment ───────────────────────────────────────────────────────

/// Append a `_distance: Float32` column to each batch.
fn attach_distances(
    batches: Vec<RecordBatch>,
    key_col_idx: usize,
    key_to_dist: &HashMap<u64, f32>,
    out_schema: &SchemaRef,
) -> Result<Vec<RecordBatch>> {
    batches
        .into_iter()
        .filter(|b| b.num_rows() > 0)
        .map(|batch| {
            let key_col = batch.column(key_col_idx);
            let keys = extract_keys_as_u64(key_col.as_ref())?;

            let distances: Float32Array = keys
                .iter()
                .map(|k| k.and_then(|key| key_to_dist.get(&key).copied()))
                .collect();

            let mut cols: Vec<Arc<dyn Array>> = batch.columns().to_vec();
            cols.push(Arc::new(distances));

            RecordBatch::try_new(out_schema.clone(), cols)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
        })
        .collect()
}
