// planner.rs — USearchQueryPlanner + USearchExecPlanner + USearchExec.
//
// The extension planner translates USearchNode (logical) into USearchExec
// (physical).  Three execution paths depending on whether the node carries
// absorbed WHERE-clause filters and the selectivity of those filters:
//
// ── No filters ───────────────────────────────────────────────────────────────
//   1. index.search() → (keys, distances)
//   2. lookup_provider.fetch_by_keys() → exactly those k rows, O(k)
//   3. Attach _distance column.
//
// ── With filters, high selectivity (> threshold) ─────────────────────────────
//   1. Pre-scan: CoalescePartitionsExec → FilterExec → DataSourceExec
//      (_key + filter cols only). Collect valid_keys from all partitions.
//   2. selectivity = valid_keys.len() / index.size()
//   3. filtered_search(query, k, |key| valid_keys.contains(key))
//   4. lookup_provider.fetch_by_keys() → O(k) rows. Attach _distance.
//
// ── With filters, low selectivity (≤ threshold) — index-get ──────────────────
//   1. Pre-scan: same as above, collect valid_keys and compute selectivity.
//   2. index.get(key) for each valid_key → compute distances → top-k heap.
//   3. lookup_provider.fetch_by_keys() → O(k) rows. Attach _distance.
//
// All I/O is deferred to USearchExec::execute() — plan_extension is purely
// structural (validate registry, compile PhysicalExprs, build scan plans).
//
// The Sort node is kept in the logical plan so DataFusion handles ordering
// by _distance / dist alias.

use std::any::Any;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_expr::{
    EquivalenceProperties, PhysicalExpr, conjunction, create_physical_expr,
};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};

use futures::StreamExt;
use usearch::ScalarKind;

use tracing::Instrument;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::Expr;

use crate::lookup::extract_keys_as_u64;
use crate::node::{DistanceType, USearchNode};
use crate::registry::VectorIndexResolver;

/// Strip table qualifiers from column references so expressions can be
/// resolved against an unqualified Arrow schema.  Mirrors the pattern in
/// DataFusion's own `physical_planner.rs::strip_column_qualifiers`.
fn strip_column_qualifier(expr: &Expr) -> Expr {
    match expr.clone().transform(|e| match &e {
        Expr::Column(col) if col.relation.is_some() => Ok(Transformed::yes(Expr::Column(
            datafusion::common::Column::new_unqualified(col.name.clone()),
        ))),
        _ => Ok(Transformed::no(e)),
    }) {
        Ok(t) => t.data,
        Err(_) => expr.clone(),
    }
}

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
    pub fn new(registry: Arc<dyn VectorIndexResolver>) -> Self {
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
    registry: Arc<dyn VectorIndexResolver>,
}

impl USearchExecPlanner {
    pub fn new(registry: Arc<dyn VectorIndexResolver>) -> Self {
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

        // Async: bind a query-stable registry entry during planning.
        let registered = self.registry.prepare(&node.table_name).await?;

        let exec_props = session_state.execution_props();

        // For the filtered path, build a pre-scan plan:
        //   CoalescePartitionsExec → FilterExec → DataSourceExec
        // DataSourceExec may have multiple partitions (file groups); FilterExec
        // evaluates the predicate per partition; CoalescePartitionsExec merges
        // all partitions into a single stream of matching rows.
        // DataFusion's physical optimizer pushes the predicate from FilterExec
        // into the Parquet reader for row group / bloom / page index pruning.
        let provider_scan = if !node.filters.is_empty() {
            let scan_schema = registered.scan_provider.schema();

            // Pre-scan projection: _key + columns referenced by filters.
            // Only these are needed — _key to collect valid keys, and filter
            // columns for predicate evaluation. Reading anything else wastes I/O.
            let filter_col_names: HashSet<&str> = node
                .filters
                .iter()
                .flat_map(|f| f.column_refs())
                .map(|c| c.name.as_str())
                .collect();
            let key_col_idx = scan_schema.index_of(&registered.key_col).map_err(|_| {
                DataFusionError::Execution(format!(
                    "USearchExec: key column '{}' not found in scan provider schema",
                    registered.key_col
                ))
            })?;
            let scalar_projection: Vec<usize> = (0..scan_schema.fields().len())
                .filter(|&i| {
                    i == key_col_idx
                        || filter_col_names.contains(scan_schema.field(i).name().as_str())
                })
                .collect();

            // Don't pass filters to scan() — FilterExec handles filtering, and
            // DataFusion's physical optimizer pushes it into the Parquet reader
            // for row group / bloom / page index pruning.
            let data_source = registered
                .scan_provider
                .scan(session_state, Some(&scalar_projection), &[], None)
                .await?;

            // Compile physical filters against the projected schema and wrap
            // in a FilterExec. Column qualifiers are stripped because the
            // projected schema (from Arrow Schema) is unqualified.
            let proj_schema = data_source.schema();
            let proj_df_schema =
                datafusion::common::DFSchema::try_from(proj_schema.as_ref().clone())?;
            let phys_filters: Vec<Arc<dyn PhysicalExpr>> = node
                .filters
                .iter()
                .map(|f| {
                    let unqualified = strip_column_qualifier(f);
                    create_physical_expr(&unqualified, &proj_df_schema, exec_props)
                })
                .collect::<Result<_>>()?;
            let predicate = conjunction(phys_filters);
            let filtered: Arc<dyn ExecutionPlan> =
                Arc::new(FilterExec::try_new(predicate, data_source)?);

            // Merge all partitions into a single stream so the pre-scan
            // collects valid keys from the entire dataset, not just one
            // partition's file group.
            let coalesced: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(filtered));

            Some(coalesced)
        } else {
            None
        };

        Ok(Some(Arc::new(USearchExec::new(SearchParams {
            table_name: node.table_name.clone(),
            registered,
            query_vec: node.query_vec_f64(),
            k: node.k,
            distance_type: node.distance_type.clone(),
            has_filters: !node.filters.is_empty(),
            provider_scan,
        }))))
    }
}

// ── Search parameters ─────────────────────────────────────────────────────────

/// All parameters needed to run a USearch query, cloned cheaply into execute().
#[derive(Clone)]
struct SearchParams {
    table_name: String,
    registered: Arc<crate::registry::RegisteredTable>,
    query_vec: Vec<f64>,
    k: usize,
    distance_type: DistanceType,
    /// Whether the query has WHERE-clause filters. Used to choose between the
    /// unfiltered HNSW path and the adaptive filtered path.
    has_filters: bool,
    /// Pre-planned provider scan for the filtered path (_key + filter cols only).
    /// Used for selectivity estimation. None for the unfiltered path.
    provider_scan: Option<Arc<dyn ExecutionPlan>>,
}

impl fmt::Debug for SearchParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchParams")
            .field("table_name", &self.table_name)
            .field("k", &self.k)
            .field("has_filters", &self.has_filters)
            .field("schema", &self.registered.schema)
            .field("key_col", &self.registered.key_col)
            .field("scalar_kind", &self.registered.scalar_kind)
            .field(
                "brute_force_threshold",
                &self.registered.config.brute_force_selectivity_threshold,
            )
            .field(
                "provider_scan",
                &self.provider_scan.as_ref().map(|_| "Some(..)"),
            )
            .finish_non_exhaustive()
    }
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
            EquivalenceProperties::new(params.registered.schema.clone()),
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
            "USearchExec: table={}, k={}, filtered={}",
            self.params.table_name, self.params.k, self.params.has_filters
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
        match self.params.provider_scan {
            Some(ref scan) => vec![scan],
            None => vec![],
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let expected = self.children().len();
        if children.len() != expected {
            return Err(DataFusionError::Internal(format!(
                "USearchExec: expected {expected} children, got {}",
                children.len()
            )));
        }
        let mut params = self.params.clone();
        if params.provider_scan.is_some() {
            params.provider_scan = Some(children.into_iter().next().unwrap());
        }
        Ok(Arc::new(USearchExec::new(params)))
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
            self.params.registered.schema.clone(),
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
        usearch.has_filters = params.has_filters,
    )
)]
async fn usearch_execute(
    params: SearchParams,
    task_ctx: Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    let registered = params.registered.clone();

    if !params.has_filters {
        // ── Unfiltered path ───────────────────────────────────────────────
        let matches = {
            let _span = tracing::info_span!(
                "usearch_hnsw_search",
                usearch.k = params.k,
                usearch.dims = params.query_vec.len(),
            )
            .entered();
            usearch_search(
                &registered.index,
                &params.query_vec,
                params.k,
                registered.scalar_kind,
            )?
        };

        if matches.keys.is_empty() {
            return Ok(vec![]);
        }

        let key_to_dist: HashMap<u64, f32> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .map(|(&k, &d)| (k, d))
            .collect();

        let fetch_keys_count = matches.keys.len();
        let data_batches = async {
            registered
                .lookup_provider
                .fetch_by_keys(&matches.keys, &registered.key_col, None)
                .await
        }
        .instrument(tracing::info_span!(
            "usearch_sqlite_fetch",
            usearch.fetch_keys = fetch_keys_count,
        ))
        .await?;

        let key_col_idx = provider_key_col_idx(&registered)?;
        let _span = tracing::info_span!("usearch_attach_distances").entered();
        attach_distances(data_batches, key_col_idx, &key_to_dist, &registered.schema)
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
        usearch.has_filters = params.has_filters,
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

    // ── Phase 1: Pre-scan for selectivity estimation ───────────────────────
    // The scan_plan is CoalescePartitionsExec → FilterExec → DataSourceExec,
    // so execute(0) yields already-filtered rows from all partitions.
    let mut stream = scan_plan.execute(0, task_ctx.clone())?;
    let mut valid_keys: HashSet<u64> = HashSet::new();

    let scan_span = tracing::info_span!("usearch_pre_scan", usearch.table = %params.table_name);
    async {
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let keys = extract_keys_as_u64(batch.column(scan_key_col_idx).as_ref())?;
            for key in keys.into_iter().flatten() {
                valid_keys.insert(key);
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
    let threshold = registered.config.brute_force_selectivity_threshold;

    let path = if selectivity <= threshold {
        "index-get"
    } else {
        "filtered_search"
    };
    tracing::Span::current().record("usearch.valid_rows", valid_keys.len());
    tracing::Span::current().record("usearch.total_rows", total);
    tracing::Span::current().record("usearch.selectivity", selectivity);
    tracing::Span::current().record("usearch.path", path);

    if selectivity <= threshold {
        // ── Low-selectivity: retrieve vectors from USearch index ─────────
        // The index stores vectors alongside the graph. Retrieve them by key,
        // compute exact distances, keep top-k, then fetch result rows from
        // the lookup provider.  This avoids the expensive full Parquet scan
        // that the previous parquet-native path required.
        let top_keys = {
            let _span = tracing::info_span!(
                "usearch_index_get_distances",
                usearch.valid_keys = valid_keys.len(),
            )
            .entered();
            index_get_top_k(
                &registered.index,
                &valid_keys,
                &params.query_vec,
                params.k,
                registered.scalar_kind,
                &params.distance_type,
            )?
        };

        if top_keys.is_empty() {
            tracing::Span::current().record("usearch.result_count", 0usize);
            return Ok(vec![]);
        }

        let fetch_keys: Vec<u64> = top_keys.iter().map(|&(k, _)| k).collect();
        let key_to_dist: HashMap<u64, f32> = top_keys.into_iter().collect();

        let fetch_keys_count = fetch_keys.len();
        let data_batches = async {
            registered
                .lookup_provider
                .fetch_by_keys(&fetch_keys, &registered.key_col, None)
                .await
        }
        .instrument(tracing::info_span!(
            "usearch_sqlite_fetch",
            usearch.fetch_keys = fetch_keys_count,
        ))
        .await?;

        let result_batches = {
            let _span = tracing::info_span!("usearch_attach_distances").entered();
            attach_distances(
                data_batches,
                lookup_key_col_idx,
                &key_to_dist,
                &registered.schema,
            )?
        };

        tracing::Span::current().record(
            "usearch.result_count",
            result_batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        );
        Ok(result_batches)
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

        let fetch_keys_count = matches.keys.len();
        let data_batches = async {
            registered
                .lookup_provider
                .fetch_by_keys(&matches.keys, &registered.key_col, None)
                .await
        }
        .instrument(tracing::info_span!(
            "usearch_sqlite_fetch",
            usearch.fetch_keys = fetch_keys_count,
        ))
        .await?;

        let result_batches = {
            let _span = tracing::info_span!("usearch_attach_distances").entered();
            attach_distances(
                data_batches,
                lookup_key_col_idx,
                &key_to_dist,
                &registered.schema,
            )?
        };

        tracing::Span::current().record(
            "usearch.result_count",
            result_batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        );
        Ok(result_batches)
    }
}

// ── USearch dispatch helpers ──────────────────────────────────────────────────

/// Call `index.search` with the native scalar type appropriate for the column.
/// Converts the usearch error into a `DataFusionError::Execution`.
pub(crate) fn usearch_search(
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

/// Retrieve vectors from the USearch index for each valid key, compute exact
/// distances against the query vector, and return the top-k (key, distance) pairs.
///
/// This is the low-selectivity path: when few rows pass the WHERE filter, it is
/// cheaper to fetch vectors by key from the index (O(1) per key) than to scan the
/// entire Parquet vector column.
///
/// For `F64` scalar kind, vectors are retrieved and distances computed in f64.
/// For all other kinds (F32, F16, BF16, I8, B1), vectors are retrieved as f32
/// (USearch dequantizes internally) and distances computed in f32.
fn index_get_top_k(
    index: &usearch::Index,
    valid_keys: &HashSet<u64>,
    query_f64: &[f64],
    k: usize,
    scalar_kind: ScalarKind,
    dist_type: &DistanceType,
) -> Result<Vec<(u64, f32)>> {
    let dim = index.dimensions();
    let mut heap: BinaryHeap<ScoredKey> = BinaryHeap::with_capacity(k + 1);

    match scalar_kind {
        ScalarKind::F64 => {
            let mut buf = vec![0.0f64; dim];
            for &key in valid_keys {
                let n = index
                    .get(key, &mut buf)
                    .map_err(|e| DataFusionError::Execution(format!("index.get({key}): {e}")))?;
                if n == 0 {
                    continue; // key not found in index (e.g. null vector was skipped during build)
                }
                let dist = compute_raw_distance_f64(&buf, query_f64, dist_type);
                if dist.is_nan() {
                    continue;
                }
                heap.push(ScoredKey {
                    distance: dist,
                    key,
                });
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        _ => {
            let query_f32: Vec<f32> = query_f64.iter().map(|&v| v as f32).collect();
            let mut buf = vec![0.0f32; dim];
            for &key in valid_keys {
                let n = index
                    .get(key, &mut buf)
                    .map_err(|e| DataFusionError::Execution(format!("index.get({key}): {e}")))?;
                if n == 0 {
                    continue;
                }
                let dist = compute_raw_distance_f32(&buf, &query_f32, dist_type);
                if dist.is_nan() {
                    continue;
                }
                heap.push(ScoredKey {
                    distance: dist,
                    key,
                });
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
    }

    let mut result: Vec<(u64, f32)> = heap
        .into_vec()
        .into_iter()
        .map(|s| (s.key, s.distance))
        .collect();
    result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(result)
}

/// A key with its computed distance, for the top-k heap.
struct ScoredKey {
    distance: f32,
    key: u64,
}

impl PartialEq for ScoredKey {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for ScoredKey {}
impl PartialOrd for ScoredKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Less)
    }
}

fn compute_raw_distance_f32(v: &[f32], q: &[f32], dist_type: &DistanceType) -> f32 {
    match dist_type {
        DistanceType::L2 => v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(),
        DistanceType::Cosine => {
            let dot: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
            let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_q: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = norm_v * norm_q;
            if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
        }
        DistanceType::NegativeDot => -v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>(),
    }
}

fn compute_raw_distance_f64(v: &[f64], q: &[f64], dist_type: &DistanceType) -> f32 {
    let d = match dist_type {
        DistanceType::L2 => v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f64>(),
        DistanceType::Cosine => {
            let dot: f64 = v.iter().zip(q).map(|(a, b)| a * b).sum();
            let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_q: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            let denom = norm_v * norm_q;
            if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
        }
        DistanceType::NegativeDot => -v.iter().zip(q).map(|(a, b)| a * b).sum::<f64>(),
    };
    d as f32
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract the distance from a single row of a vector column.
///
/// Index of the key column in the lookup provider schema.
pub(crate) fn provider_key_col_idx(registered: &crate::registry::RegisteredTable) -> Result<usize> {
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
pub(crate) fn attach_distances(
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
