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
use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use datafusion_vector_search_ext::{
    HashKeyProvider, USearchQueryPlanner, USearchRegistry, USearchTableConfig, register_all,
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

// ═══════════════════════════════════════════════════════════════════════════════
// Registration validation
// ═══════════════════════════════════════════════════════════════════════════════

/// Registration must fail when scan_provider schema is missing the key column.
#[tokio::test]
async fn reg_scan_provider_missing_key_col_errors() {
    // scan_provider schema: only "label" and "vector" — no "id".
    let scan_schema = Arc::new(Schema::new(vec![
        Field::new("label", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
    ]));
    let scan_provider =
        Arc::new(HashKeyProvider::try_new(scan_schema, vec![], "label").expect("HashKeyProvider"));

    // lookup_provider has "id".
    let lookup_schema = exec_schema();
    let lookup_provider =
        Arc::new(HashKeyProvider::try_new(lookup_schema, vec![], "id").expect("HashKeyProvider"));

    let reg = USearchRegistry::new();
    let result = reg.add(
        "test::vector",
        make_populated_index(),
        scan_provider,
        lookup_provider,
        "id",
        MetricKind::L2sq,
        ScalarKind::F32,
    );
    assert!(
        result.is_err(),
        "registration must fail when scan_provider lacks key column"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("scan provider"),
        "error must mention scan provider: {msg}"
    );
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

// ═══════════════════════════════════════════════════════════════════════════════
// Parquet-native path — low-selectivity execution
//
// These tests force the parquet-native path by setting
// brute_force_selectivity_threshold = 1.0, so ALL filtered queries bypass
// HNSW+SQLite. The lookup_provider schema excludes the vector column,
// matching the real Parquet+SQLite deployment.
// ═══════════════════════════════════════════════════════════════════════════════

/// Schema for the lookup provider — no vector column.
fn lookup_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("label", DataType::Utf8, false),
    ]))
}

/// Build context that forces the parquet-native path for all filtered queries.
async fn make_parquet_native_ctx(reg_key: &str) -> SessionContext {
    let schema = exec_schema();
    let batch = test_batch(&schema);

    // scan_provider: full schema including vector column (simulates Parquet)
    let scan_provider = Arc::new(
        HashKeyProvider::try_new(schema.clone(), vec![batch.clone()], "id")
            .expect("scan HashKeyProvider"),
    );

    // lookup_provider: no vector column (simulates SQLite)
    let lookup_batch = {
        let ids = batch.column(0).clone();
        let labels = batch.column(1).clone();
        RecordBatch::try_new(lookup_schema(), vec![ids, labels]).expect("lookup batch")
    };
    let lookup_provider = Arc::new(
        HashKeyProvider::try_new(lookup_schema(), vec![lookup_batch], "id")
            .expect("lookup HashKeyProvider"),
    );

    let reg = USearchRegistry::new();
    reg.add_with_config(
        reg_key,
        make_populated_index(),
        scan_provider,
        lookup_provider,
        "id",
        MetricKind::L2sq,
        ScalarKind::F32,
        USearchTableConfig {
            brute_force_selectivity_threshold: 1.0, // force parquet-native for all filters
            ..Default::default()
        },
    )
    .expect("reg.add_with_config");
    let registry = reg.into_arc();

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(USearchQueryPlanner::new(registry.clone())))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_all(&ctx, registry).expect("register_all");

    // Register scan_provider as the table (so DataFusion can resolve column refs).
    let table_provider = Arc::new(
        HashKeyProvider::try_new(exec_schema(), vec![test_batch(&exec_schema())], "id")
            .expect("table HashKeyProvider"),
    );
    ctx.register_table("items", table_provider)
        .expect("register_table");
    ctx
}

/// Parquet-native: WHERE excludes rows, results must respect filter and distance ordering.
#[tokio::test]
async fn exec_parquet_native_where_clause() {
    let ctx = make_parquet_native_ctx("items::vector").await;
    let sql = format!(
        "SELECT id FROM items WHERE label != 'alpha' ORDER BY l2_distance(vector, {Q}) ASC LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    assert!(ids.contains(&2), "row 2 (beta) must appear; got {ids:?}");
    assert!(ids.contains(&3), "row 3 (gamma) must appear; got {ids:?}");
    assert!(
        !ids.contains(&1),
        "row 1 (alpha) must be filtered out; got {ids:?}"
    );
    assert!(
        !ids.contains(&4),
        "row 4 (alpha) must be filtered out; got {ids:?}"
    );
}

/// Parquet-native: equality filter, verify distance ordering.
#[tokio::test]
async fn exec_parquet_native_equality_filter() {
    let ctx = make_parquet_native_ctx("items::vector").await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS dist FROM items WHERE label = 'alpha' ORDER BY dist ASC LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    // Rows 1 and 4 match label='alpha'. Row 1 is closer (L2sq=0 vs L2sq=2).
    assert_eq!(
        ids[0], 1,
        "closest alpha row must be row 1 (dist=0)\nids: {ids:?}"
    );
    assert!(ids.contains(&4), "row 4 must be in results\nids: {ids:?}");
}

/// Parquet-native: LIMIT < matching rows, verifies top-k heap eviction works.
#[tokio::test]
async fn exec_parquet_native_limit_fewer_than_matches() {
    let ctx = make_parquet_native_ctx("items::vector").await;
    let sql = format!(
        "SELECT id FROM items WHERE label != 'alpha' ORDER BY l2_distance(vector, {Q}) ASC LIMIT 1"
    );
    let ids = collect_ids(&ctx, &sql).await;
    assert_eq!(ids.len(), 1, "exactly 1 result expected; got {ids:?}");
    // Rows 2 and 3 both have L2sq=2 from query; either is valid, but only 1 returned.
    assert!(
        ids[0] == 2 || ids[0] == 3,
        "must be row 2 or 3; got {ids:?}"
    );
}

/// Parquet-native: WHERE filters all rows, must return empty.
#[tokio::test]
async fn exec_parquet_native_where_no_matches() {
    let ctx = make_parquet_native_ctx("items::vector").await;
    let sql = format!(
        "SELECT id FROM items WHERE label = 'nonexistent' ORDER BY l2_distance(vector, {Q}) ASC LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    assert!(ids.is_empty(), "no rows should match; got {ids:?}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Numeric regression — l2_distance must return L2sq (no sqrt)
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Split-provider tests — lookup_provider WITHOUT vector column
//
// In production, lookup_provider (SQLite) does NOT have the vector column.
// These tests verify that the USearch optimized path fires and works correctly
// when selecting specific columns (not SELECT *).
// ═══════════════════════════════════════════════════════════════════════════════

/// Build context with split providers (scan has vector, lookup doesn't)
/// and default brute_force threshold.
async fn make_split_provider_ctx(reg_key: &str) -> SessionContext {
    let schema = exec_schema();
    let batch = test_batch(&schema);

    // scan_provider: full schema including vector column (simulates Parquet)
    let scan_provider = Arc::new(
        HashKeyProvider::try_new(schema.clone(), vec![batch.clone()], "id")
            .expect("scan HashKeyProvider"),
    );

    // lookup_provider: no vector column (simulates SQLite)
    let lookup_batch = {
        let ids = batch.column(0).clone();
        let labels = batch.column(1).clone();
        RecordBatch::try_new(lookup_schema(), vec![ids, labels]).expect("lookup batch")
    };
    let lookup_provider = Arc::new(
        HashKeyProvider::try_new(lookup_schema(), vec![lookup_batch], "id")
            .expect("lookup HashKeyProvider"),
    );

    let reg = USearchRegistry::new();
    reg.add(
        reg_key,
        make_populated_index(),
        scan_provider,
        lookup_provider,
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

    let table_provider = Arc::new(
        HashKeyProvider::try_new(exec_schema(), vec![test_batch(&exec_schema())], "id")
            .expect("table HashKeyProvider"),
    );
    ctx.register_table("items", table_provider)
        .expect("register_table");
    ctx
}

/// SELECT specific columns (no vector) with distance UDF — must use USearch path.
/// This is the exact pattern that fails in production while SELECT * works.
#[tokio::test]
async fn exec_split_provider_select_specific_columns() {
    let ctx = make_split_provider_ctx("items::vector").await;
    let sql =
        format!("SELECT id, l2_distance(vector, {Q}) AS dist FROM items ORDER BY dist ASC LIMIT 2");
    let ids = collect_ids(&ctx, &sql).await;
    assert_eq!(ids[0], 1, "closest must be row 1\nids: {ids:?}");
    assert_eq!(ids.len(), 2, "expected 2 results; got {ids:?}");
}

/// SELECT specific columns without projecting the distance expression.
/// This is the production shape behind `vector_distance(...)`.
#[tokio::test]
async fn exec_split_provider_order_by_udf_direct() {
    let ctx = make_split_provider_ctx("items::vector").await;
    let sql = format!("SELECT id FROM items ORDER BY l2_distance(vector, {Q}) ASC LIMIT 2");
    let ids = collect_ids(&ctx, &sql).await;
    assert_eq!(ids[0], 1, "closest must be row 1\nids: {ids:?}");
    assert_eq!(ids.len(), 2, "expected 2 results; got {ids:?}");
}

/// SELECT * with distance UDF — should fall back to UDF brute-force
/// (since vector column is not in lookup provider schema).
#[tokio::test]
async fn exec_split_provider_select_star() {
    let ctx = make_split_provider_ctx("items::vector").await;
    let sql =
        format!("SELECT *, l2_distance(vector, {Q}) AS dist FROM items ORDER BY dist ASC LIMIT 2");
    let df = ctx.sql(&sql).await.expect("sql");
    let batches = df.collect().await.expect("collect");
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2, "expected 2 results");
}

/// SELECT specific columns with fully qualified table name.
#[tokio::test]
async fn exec_split_provider_qualified_select_specific() {
    let ctx = make_split_provider_ctx("datafusion::public::items::vector").await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS dist FROM datafusion.public.items ORDER BY dist ASC LIMIT 2"
    );
    let ids = collect_ids(&ctx, &sql).await;
    assert_eq!(ids[0], 1, "closest must be row 1\nids: {ids:?}");
}

/// negative_dot_product with split providers and IP metric — mirrors production setup.
#[tokio::test]
async fn exec_split_provider_negative_dot_product() {
    let schema = exec_schema();
    let batch = test_batch(&schema);

    let scan_provider = Arc::new(
        HashKeyProvider::try_new(schema.clone(), vec![batch.clone()], "id")
            .expect("scan HashKeyProvider"),
    );

    let lookup_batch = {
        let ids = batch.column(0).clone();
        let labels = batch.column(1).clone();
        RecordBatch::try_new(lookup_schema(), vec![ids, labels]).expect("lookup batch")
    };
    let lookup_provider = Arc::new(
        HashKeyProvider::try_new(lookup_schema(), vec![lookup_batch], "id")
            .expect("lookup HashKeyProvider"),
    );

    // Build IP-metric index
    let opts = IndexOptions {
        dimensions: 4,
        metric: MetricKind::IP,
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

    let reg = USearchRegistry::new();
    reg.add(
        "items::vector",
        index,
        scan_provider,
        lookup_provider,
        "id",
        MetricKind::IP,
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
    let table_provider = Arc::new(
        HashKeyProvider::try_new(exec_schema(), vec![test_batch(&exec_schema())], "id")
            .expect("table HashKeyProvider"),
    );
    ctx.register_table("items", table_provider)
        .expect("register_table");

    // This is the exact pattern that fails: SELECT specific_cols, negative_dot_product, ORDER BY alias
    let sql = "SELECT id, negative_dot_product(vector, ARRAY[1.0::float, 0.0::float, 0.0::float, 0.0::float]) AS dist FROM items ORDER BY dist ASC LIMIT 2";
    let ids = collect_ids(&ctx, sql).await;
    assert_eq!(ids[0], 1, "closest must be row 1\nids: {ids:?}");
}

/// 768-dim negative_dot_product with split providers — reproduces production query pattern.
#[tokio::test]
async fn exec_split_provider_768dim_negative_dot_product() {
    let dim = 768i32;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("label", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]));

    let ids_arr = UInt64Array::from(vec![1u64, 2, 3, 4]);
    let labels_arr = StringArray::from(vec!["a", "b", "c", "d"]);
    let vecs: Vec<Vec<f32>> = (0..4)
        .map(|row| {
            (0..dim as usize)
                .map(|i| ((row * dim as usize + i) as f32) * 0.001)
                .collect()
        })
        .collect();
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
    for v in &vecs {
        builder.values().append_slice(v);
        builder.append(true);
    }
    let vector_col: FixedSizeListArray = builder.finish();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids_arr),
            Arc::new(labels_arr),
            Arc::new(vector_col),
        ],
    )
    .unwrap();

    let scan_provider =
        Arc::new(HashKeyProvider::try_new(schema.clone(), vec![batch.clone()], "id").unwrap());
    let lookup_batch = RecordBatch::try_new(
        lookup_schema(),
        vec![batch.column(0).clone(), batch.column(1).clone()],
    )
    .unwrap();
    let lookup_provider =
        Arc::new(HashKeyProvider::try_new(lookup_schema(), vec![lookup_batch], "id").unwrap());

    let opts = IndexOptions {
        dimensions: dim as usize,
        metric: MetricKind::IP,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    let index = Arc::new(Index::new(&opts).unwrap());
    index.reserve(4).unwrap();
    for (row, key) in vecs.iter().zip([1u64, 2, 3, 4]) {
        index.add(key, row.as_slice()).unwrap();
    }

    let reg = USearchRegistry::new();
    reg.add(
        "items::vector",
        index,
        scan_provider,
        lookup_provider,
        "id",
        MetricKind::IP,
        ScalarKind::F32,
    )
    .unwrap();
    let registry = reg.into_arc();

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(USearchQueryPlanner::new(registry.clone())))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_all(&ctx, registry).unwrap();

    let table_provider = Arc::new(HashKeyProvider::try_new(schema, vec![batch], "id").unwrap());
    ctx.register_table("items", table_provider).unwrap();

    // Build 768-element query array
    let query_arr: Vec<String> = (0..dim)
        .map(|i| format!("{:.6}", i as f64 * 0.001))
        .collect();
    let query_str = query_arr.join(",");
    let sql = format!(
        "SELECT id, negative_dot_product(vector, ARRAY[{}]) AS dist FROM items ORDER BY dist ASC LIMIT 2",
        query_str
    );

    let df = ctx.sql(&sql).await.expect("sql failed");
    let batches = df.collect().await.expect("collect failed");
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2, "expected 2 results");
}

/// l2_distance must return squared L2, not actual L2.
/// Row 1 = [1,0,0,0], query = [1,0,0,0] → L2sq = 0.0
/// Row 2 = [0,1,0,0], query = [1,0,0,0] → L2sq = 2.0 (L2 would be ~1.414)
#[tokio::test]
async fn exec_l2_distance_returns_l2sq() {
    let ctx = make_exec_ctx("items::vector").await;
    let sql =
        format!("SELECT id, l2_distance(vector, {Q}) AS dist FROM items ORDER BY dist ASC LIMIT 4");
    let df = ctx.sql(&sql).await.expect("sql");
    let batches = df.collect().await.expect("collect");

    let mut dists: Vec<(u64, f32)> = vec![];
    for batch in &batches {
        let id_idx = batch.schema().index_of("id").unwrap();
        let dist_idx = batch.schema().index_of("dist").unwrap();
        let ids = batch
            .column(id_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let ds = batch
            .column(dist_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            dists.push((ids.value(i), ds.value(i)));
        }
    }

    // Row 1: exact match → 0.0
    let row1 = dists
        .iter()
        .find(|(id, _)| *id == 1)
        .expect("row 1 missing");
    assert!(
        (row1.1 - 0.0).abs() < 1e-6,
        "row 1 distance must be 0.0 (L2sq); got {}",
        row1.1
    );

    // Row 2: [0,1,0,0] vs [1,0,0,0] → L2sq = 2.0, NOT sqrt(2) ≈ 1.414
    let row2 = dists
        .iter()
        .find(|(id, _)| *id == 2)
        .expect("row 2 missing");
    assert!(
        (row2.1 - 2.0).abs() < 1e-6,
        "row 2 distance must be 2.0 (L2sq), not {:.4} (would be ~1.414 if L2)",
        row2.1
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dimension mismatch — UDF must reject mismatched query vectors
// ═══════════════════════════════════════════════════════════════════════════════

/// Dimension mismatch must error — optimizer path (USearch) catches it.
#[tokio::test]
async fn udf_dimension_mismatch_fewer() {
    let ctx = make_exec_ctx("items::vector").await;
    // Column is 4-dim, query is 3-dim
    let sql = "SELECT id, l2_distance(vector, ARRAY[1.0::float, 0.0::float, 0.0::float]) AS dist FROM items ORDER BY dist ASC LIMIT 2";
    let err = ctx
        .sql(sql)
        .await
        .expect("sql")
        .collect()
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("must match"),
        "expected dimension mismatch error, got: {msg}"
    );
}

/// Dimension mismatch must error — optimizer path (USearch) catches it.
#[tokio::test]
async fn udf_dimension_mismatch_more() {
    let ctx = make_exec_ctx("items::vector").await;
    // Column is 4-dim, query is 5-dim
    let sql = "SELECT id, l2_distance(vector, ARRAY[1.0::float, 0.0::float, 0.0::float, 0.0::float, 0.0::float]) AS dist FROM items ORDER BY dist ASC LIMIT 2";
    let err = ctx
        .sql(sql)
        .await
        .expect("sql")
        .collect()
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("must match"),
        "expected dimension mismatch error, got: {msg}"
    );
}

/// SELECT * with mismatched dimensions must also error (not silently truncate).
/// This is the key test: SELECT * bypasses the optimizer (vector column not in
/// lookup schema), so the UDF brute-force path runs. Before the fix, zip()
/// silently truncated and returned wrong results.
#[tokio::test]
async fn udf_dimension_mismatch_select_star() {
    let ctx = make_split_provider_ctx("items::vector").await;
    // Column is 4-dim, query is 3-dim. SELECT * falls back to UDF path.
    let sql = "SELECT *, l2_distance(vector, ARRAY[1.0::float, 0.0::float, 0.0::float]) AS dist FROM items ORDER BY dist ASC LIMIT 2";
    let err = ctx
        .sql(sql)
        .await
        .expect("sql")
        .collect()
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("must match"),
        "expected dimension mismatch error, got: {msg}"
    );
}
