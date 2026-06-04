// tests/orderby_distance_trimmed.rs — Regression tests for the "trimmed"
// k-NN shape: `ORDER BY distance_fn(vector, lit)` where the distance is NOT in
// the SELECT list and the query has a WHERE clause.
//
// With a Filter present, DataFusion materializes the raw vector column in an
// intermediate projection between the Sort and the Filter (it is needed to
// evaluate the inline ORDER BY expression), then trims it with an outer
// projection:
//
//   Projection: id                      ← real output (no vector)
//     Sort: l2_distance(vector, lit)
//       Projection: id, vector          ← vector materialized only for the Sort
//         Filter: label = '…'
//           TableScan
//
// The Sort-anchored match judges producibility on the INNER projection — which
// contains the vector the node cannot produce — and would wrongly fall back.
// The Projection-anchored match must recognize this shape and judge
// producibility on the OUTER projection instead (issue: ORDER-BY-only distance
// silently losing the index whenever a WHERE clause is present).
//
// Unlike tests/vector_col_projection.rs, the fixtures here use a ducklake-style
// addressing key — `rowid: Int64` — rather than the parquet-style
// `_key: UInt64`, so the key-column-agnostic path is covered too.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use datafusion_vector_search_ext::{HashKeyProvider, USearchNode, USearchRegistry, register_all};

/// The user-visible table: addressing key absent, vector column present.
fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
    ]))
}

/// Sidecar/lookup schema: ducklake-style `rowid` key + non-vector columns.
/// The vector column is excluded — exactly as the SQLite sidecar stores it.
fn lookup_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("rowid", DataType::Int64, false),
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
    ]))
}

/// Scan-provider schema: full column set including the vector, with the
/// `rowid` key — mirrors the snapshot-pinned DuckLake scan provider.
fn scan_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
        Field::new("rowid", DataType::Int64, false),
    ]))
}

fn make_index() -> Arc<Index> {
    let options = IndexOptions {
        dimensions: 4,
        metric: MetricKind::L2sq,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    Arc::new(Index::new(&options).expect("usearch Index::new failed"))
}

async fn make_ctx() -> SessionContext {
    let scan_provider = Arc::new(
        HashKeyProvider::try_new(scan_schema(), vec![], "rowid")
            .expect("scan HashKeyProvider::try_new failed"),
    );
    let lookup_provider = Arc::new(
        HashKeyProvider::try_new(lookup_schema(), vec![], "rowid")
            .expect("lookup HashKeyProvider::try_new failed"),
    );

    let reg = USearchRegistry::new();
    reg.add(
        "items::embedding",
        make_index(),
        scan_provider,
        lookup_provider,
        "rowid",
        MetricKind::L2sq,
        ScalarKind::F32,
    )
    .expect("USearchRegistry::add failed");
    let registry = reg.into_arc();

    let ctx = SessionContext::default();
    register_all(&ctx, registry).expect("register_all failed");

    let table = Arc::new(
        HashKeyProvider::try_new(table_schema(), vec![], "id")
            .expect("table HashKeyProvider::try_new failed"),
    );
    ctx.register_table("items", table)
        .expect("register_table failed");
    ctx
}

fn contains_usearch_node(plan: &LogicalPlan) -> bool {
    if let LogicalPlan::Extension(ext) = plan
        && ext.node.as_any().downcast_ref::<USearchNode>().is_some()
    {
        return true;
    }
    plan.inputs().iter().any(|c| contains_usearch_node(c))
}

const Q: &str = "ARRAY[0.1, 0.2, 0.3, 0.4]";

/// The shape this file exists for: distance only in ORDER BY, WHERE present.
/// The vector appears in DataFusion's intermediate projection but not in the
/// query output — the rule must use the index.
#[tokio::test]
async fn test_bare_orderby_with_where_rewrites() {
    let ctx = make_ctx().await;
    let sql = format!(
        "SELECT id FROM items WHERE label = 'x' \
         ORDER BY l2_distance(embedding, {Q}) ASC LIMIT 2"
    );
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization must not error");
    assert!(
        contains_usearch_node(&plan),
        "vector not in output → rule must use the index despite the WHERE-induced \
         intermediate projection\nPlan: {plan:?}"
    );
}

/// Multiple non-vector output columns, same shape.
#[tokio::test]
async fn test_bare_orderby_with_where_multiple_columns_rewrites() {
    let ctx = make_ctx().await;
    let sql = format!(
        "SELECT id, label FROM items WHERE label LIKE 'x%' \
         ORDER BY l2_distance(embedding, {Q}) ASC LIMIT 5"
    );
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization must not error");
    assert!(
        contains_usearch_node(&plan),
        "all output columns producible → rule must use the index\nPlan: {plan:?}"
    );
}

/// `SELECT *` with WHERE: output includes the vector → must still fall back.
#[tokio::test]
async fn test_select_star_with_where_still_falls_back() {
    let ctx = make_ctx().await;
    let sql = format!(
        "SELECT * FROM items WHERE label = 'x' \
         ORDER BY l2_distance(embedding, {Q}) ASC LIMIT 2"
    );
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization must not error when the vector column is in the output");
    assert!(
        !contains_usearch_node(&plan),
        "vector column in output → rule must fall back, WHERE or not\nPlan: {plan:?}"
    );
}

/// Explicit vector column in the output with WHERE: must still fall back.
#[tokio::test]
async fn test_select_vector_with_where_still_falls_back() {
    let ctx = make_ctx().await;
    let sql = format!(
        "SELECT id, embedding FROM items WHERE label = 'x' \
         ORDER BY l2_distance(embedding, {Q}) ASC LIMIT 2"
    );
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization must not error when the vector column is in the output");
    assert!(
        !contains_usearch_node(&plan),
        "vector column in output → rule must fall back, WHERE or not\nPlan: {plan:?}"
    );
}

/// The canonical aliased-distance shape with WHERE keeps rewriting via the
/// Sort-anchored match (regression guard: the new Projection-anchored arm must
/// decline it cleanly and leave it to the Sort visit).
#[tokio::test]
async fn test_aliased_distance_with_where_still_rewrites() {
    let ctx = make_ctx().await;
    let sql = format!(
        "SELECT id, l2_distance(embedding, {Q}) AS dist FROM items \
         WHERE label = 'x' ORDER BY dist ASC LIMIT 2"
    );
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization failed");
    assert!(
        contains_usearch_node(&plan),
        "aliased-distance shape must keep rewriting\nPlan: {plan:?}"
    );
}
