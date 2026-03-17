// tests/execution.rs — End-to-end execution tests for USearch queries.
//
// These tests go beyond optimizer-rule matching: they build real indices with
// populated data, wire up USearchQueryPlanner, and call .collect() to verify
// that the physical execution path works correctly for every supported query
// form — including those where the broken filter-DFSchema path was hit.
//
// Schema: id: UInt64, label: Utf8, vector: FixedSizeList<f32, 4>
//
// Rows:
//   1  "alpha"   [1.0, 0.0, 0.0, 0.0]
//   2  "beta"    [0.0, 1.0, 0.0, 0.0]
//   3  "gamma"   [0.0, 0.0, 1.0, 0.0]
//   4  "alpha"   [0.0, 0.0, 0.0, 1.0]
//
// Query vector [1.0, 0.0, 0.0, 0.0] is closest to row 1 (L2sq = 0), then
// rows 2/3/4 (L2sq = 2).

use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{FixedSizeListArray, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use datafusion_vector_search_ext::{
    HashKeyProvider, USearchQueryPlanner, USearchRegistry, register_all,
};

// ── Schema & data ─────────────────────────────────────────────────────────────

fn exec_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("label", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
    ]))
}

/// Build the 4-row test RecordBatch.
fn test_batch(schema: &Arc<Schema>) -> RecordBatch {
    let ids = UInt64Array::from(vec![1u64, 2, 3, 4]);
    let labels = StringArray::from(vec!["alpha", "beta", "gamma", "alpha"]);

    let vectors: &[[f32; 4]] = &[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 4);
    for v in vectors {
        builder.values().append_slice(v);
        builder.append(true);
    }
    let vector_col: FixedSizeListArray = builder.finish();

    RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(labels), Arc::new(vector_col)],
    )
    .expect("test_batch build failed")
}

/// Build a populated L2sq USearch index from the 4 test rows.
fn make_populated_index() -> Arc<Index> {
    let opts = IndexOptions {
        dimensions: 4,
        metric: MetricKind::L2sq,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    let index = Arc::new(Index::new(&opts).expect("Index::new"));
    index.reserve(4).expect("reserve");
    let rows: &[(u64, [f32; 4])] = &[
        (1, [1.0, 0.0, 0.0, 0.0]),
        (2, [0.0, 1.0, 0.0, 0.0]),
        (3, [0.0, 0.0, 1.0, 0.0]),
        (4, [0.0, 0.0, 0.0, 1.0]),
    ];
    for &(key, ref v) in rows {
        index.add(key, v.as_slice()).expect("index.add");
    }
    index
}

// ── Context factories ─────────────────────────────────────────────────────────

/// Context with `USearchQueryPlanner` and table registered under bare name.
async fn make_exec_ctx(reg_key: &str) -> SessionContext {
    let schema = exec_schema();
    let batch = test_batch(&schema);
    let provider = Arc::new(
        HashKeyProvider::try_new(schema.clone(), vec![batch], "id")
            .expect("HashKeyProvider::try_new"),
    );
    let reg = USearchRegistry::new();
    reg.add(
        reg_key,
        make_populated_index(),
        provider.clone(),
        provider.clone(),
        "id",
        MetricKind::L2sq,
        ScalarKind::F32,
    )
    .expect("reg.add");
    let registry = reg.into_arc();

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(USearchQueryPlanner::new(registry.clone())))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_all(&ctx, registry).expect("register_all");
    ctx.register_table("items", provider)
        .expect("register_table");
    ctx
}

/// Collect ids from a query result (first UInt64 column named "id").
async fn collect_ids(ctx: &SessionContext, sql: &str) -> Vec<u64> {
    let df = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|e| panic!("sql() failed: {e}\nSQL: {sql}"));
    let batches = df
        .collect()
        .await
        .unwrap_or_else(|e| panic!("collect() failed: {e}\nSQL: {sql}"));

    let mut ids: Vec<u64> = vec![];
    for batch in &batches {
        let col_idx = batch
            .schema()
            .index_of("id")
            .expect("no 'id' column in result");
        let arr = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("id column not UInt64");
        ids.extend(arr.values());
    }
    ids
}

const Q: &str = "ARRAY[1.0::float, 0.0::float, 0.0::float, 0.0::float]";

// ═══════════════════════════════════════════════════════════════════════════════
// Basic execution — bare table name
// ═══════════════════════════════════════════════════════════════════════════════

/// Baseline: ORDER BY UDF directly, bare table name.
#[tokio::test]
async fn exec_order_by_udf_bare() {
    let ctx = make_exec_ctx("items::vector").await;
    let sql = format!("SELECT id FROM items ORDER BY l2_distance(vector, {Q}) ASC LIMIT 2");
    let ids = collect_ids(&ctx, &sql).await;
    assert_eq!(
        ids[0], 1,
        "closest to [1,0,0,0] must be row 1\nids: {ids:?}"
    );
}

/// Alias in ORDER BY, bare table.
#[tokio::test]
async fn exec_order_by_alias_bare() {
    let ctx = make_exec_ctx("items::vector").await;
    let sql =
        format!("SELECT id, l2_distance(vector, {Q}) AS dist FROM items ORDER BY dist ASC LIMIT 2");
    let ids = collect_ids(&ctx, &sql).await;
    assert_eq!(ids[0], 1, "closest must be row 1\nids: {ids:?}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// WHERE clause execution — bare table name
// ═══════════════════════════════════════════════════════════════════════════════

/// WHERE clause with ORDER BY UDF: rows 2/3/4 pass `label != 'alpha'`; row 2 is
/// closest to the query vector among those.
#[tokio::test]
async fn exec_where_clause_bare() {
    let ctx = make_exec_ctx("items::vector").await;
    let sql = format!(
        "SELECT id FROM items WHERE label != 'alpha' ORDER BY l2_distance(vector, {Q}) ASC LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    assert!(ids.contains(&2), "row 2 (beta) must appear; got {ids:?}");
    assert!(
        !ids.contains(&1),
        "row 1 (alpha) must be filtered out; got {ids:?}"
    );
    assert!(
        !ids.contains(&4),
        "row 4 (alpha) must be filtered out; got {ids:?}"
    );
}

/// WHERE clause with alias in ORDER BY, bare table.
#[tokio::test]
async fn exec_where_clause_alias_bare() {
    let ctx = make_exec_ctx("items::vector").await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS dist FROM items WHERE label = 'alpha' ORDER BY dist ASC LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    // Rows 1 and 4 match label='alpha'. Row 1 is closer.
    assert_eq!(ids[0], 1, "closest alpha row must be row 1\nids: {ids:?}");
    assert!(ids.contains(&4), "row 4 must be in results\nids: {ids:?}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fully-qualified table reference — WHERE clause filter
//
// This is the regression path: the filter predicate column reference carries the
// full catalog.schema.table qualifier; before the fix `create_physical_expr`
// failed because it was compiled against a bare DFSchema.
// ═══════════════════════════════════════════════════════════════════════════════

/// Subquery form: subquery does the filtered projection; outer query sorts on
/// the alias and applies LIMIT.  This is the exact query form from the bug report.
#[tokio::test]
async fn exec_qualified_subquery_where_order_by_alias() {
    let ctx = make_exec_ctx("datafusion::public::items::vector").await;
    let sql = format!(
        "SELECT * FROM (
            SELECT id, label, l2_distance(vector, {Q}) AS dist
            FROM datafusion.public.items
            WHERE label != 'alpha'
        ) t ORDER BY dist LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    assert!(ids.contains(&2), "row 2 (beta) must appear; got {ids:?}");
    assert!(
        !ids.contains(&1),
        "row 1 (alpha) must be filtered; got {ids:?}"
    );
}

/// Simpler form: qualified table, WHERE clause, ORDER BY alias inline.
#[tokio::test]
async fn exec_qualified_where_order_by_alias() {
    let ctx = make_exec_ctx("datafusion::public::items::vector").await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS dist FROM datafusion.public.items
         WHERE label = 'alpha' ORDER BY dist ASC LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    assert_eq!(ids[0], 1, "closest alpha row must be row 1\nids: {ids:?}");
}

/// Qualified table, WHERE clause, ORDER BY UDF directly.
#[tokio::test]
async fn exec_qualified_where_order_by_udf() {
    let ctx = make_exec_ctx("datafusion::public::items::vector").await;
    let sql = format!(
        "SELECT id FROM datafusion.public.items
         WHERE label != 'alpha' ORDER BY l2_distance(vector, {Q}) ASC LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    assert!(ids.contains(&2), "row 2 (beta) must appear; got {ids:?}");
    assert!(
        !ids.contains(&1),
        "row 1 (alpha) must be filtered; got {ids:?}"
    );
}
