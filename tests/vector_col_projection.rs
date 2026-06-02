// tests/vector_col_projection.rs — Regression tests for the case where a k-NN
// query projects the indexed vector column itself (or SELECT *).
//
// Unlike tests/optimizer_rule.rs, the lookup provider's schema here DELIBERATELY
// EXCLUDES the vector column — faithfully modelling production, where the SQLite
// sidecar stores only the addressing key + non-vector columns (the vector itself
// is never stored). The registry derives meta.schema from the lookup provider, so
// meta.schema lacks the vector column.
//
// In this configuration the rewrite cannot reproduce the vector column in its
// output. The rule must therefore decline to fire (fall back to brute-force exact
// search) rather than produce a plan whose output schema differs from the
// original — which trips DataFusion's post-optimizer invariant check.

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

/// The sidecar/lookup schema: synthetic addressing key + non-vector columns.
/// The vector column is excluded — exactly as the SQLite sidecar stores it.
fn lookup_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("_key", DataType::UInt64, false),
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
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
    // Registry's scan provider mirrors the parquet provider: full schema incl.
    // the synthetic `_key` and the vector column. Keyed on `_key`.
    let scan_provider = Arc::new(
        HashKeyProvider::try_new(
            Arc::new(Schema::new(vec![
                Field::new("_key", DataType::UInt64, false),
                Field::new("id", DataType::Int32, false),
                Field::new("label", DataType::Utf8, true),
                Field::new(
                    "embedding",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        4,
                    ),
                    false,
                ),
            ])),
            vec![],
            "_key",
        )
        .expect("scan HashKeyProvider::try_new failed"),
    );

    // Registry's lookup provider mirrors the SQLite sidecar: NO vector column.
    let lookup_provider = Arc::new(
        HashKeyProvider::try_new(lookup_schema(), vec![], "_key")
            .expect("lookup HashKeyProvider::try_new failed"),
    );

    let reg = USearchRegistry::new();
    reg.add(
        "items::embedding",
        make_index(),
        scan_provider,
        lookup_provider,
        "_key",
        MetricKind::L2sq,
        ScalarKind::F32,
    )
    .expect("USearchRegistry::add failed");
    let registry = reg.into_arc();

    let ctx = SessionContext::default();
    register_all(&ctx, registry).expect("register_all failed");

    // The SQL-visible table carries the real columns (no `_key`).
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

/// SELECT * over an indexed table: the output includes the vector column, which
/// the rewrite cannot produce. The rule must NOT fire, and optimization must
/// succeed (falling back to exact search) rather than erroring on a schema
/// mismatch.
#[tokio::test]
async fn test_select_star_with_vector_index_does_not_crash() {
    let ctx = make_ctx().await;
    let sql = format!("SELECT * FROM items ORDER BY l2_distance(embedding, {Q}) ASC LIMIT 2");
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization must not error when the vector column is in the output");
    assert!(
        !contains_usearch_node(&plan),
        "vector column in output → rule must fall back to exact search, not rewrite\nPlan: {plan:?}"
    );
}

/// Explicitly projecting the indexed vector column has the same requirement.
#[tokio::test]
async fn test_select_vector_column_does_not_crash() {
    let ctx = make_ctx().await;
    let sql =
        format!("SELECT id, embedding FROM items ORDER BY l2_distance(embedding, {Q}) ASC LIMIT 2");
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization must not error when the vector column is in the output");
    assert!(
        !contains_usearch_node(&plan),
        "vector column in output → rule must fall back to exact search, not rewrite\nPlan: {plan:?}"
    );
}

/// The canonical vector-search query (distance aliased in the SELECT, ORDER BY
/// the alias) keeps rewriting: its output columns (id, the distance) are all
/// producible from the sidecar, and the aliased distance forces a Projection
/// below the Sort so the rewrite reproduces the schema exactly.
#[tokio::test]
async fn test_aliased_distance_still_rewrites() {
    let ctx = make_ctx().await;
    let sql = format!(
        "SELECT id, l2_distance(embedding, {Q}) AS dist FROM items ORDER BY dist ASC LIMIT 2"
    );
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization failed");
    assert!(
        contains_usearch_node(&plan),
        "no vector column in output → rule must still fire\nPlan: {plan:?}"
    );
}

/// A bare projection whose ORDER BY computes the distance inline (distance not in
/// the SELECT, no Projection below the Sort) — the common "give me the nearest
/// rows' ids" query. The indexed vector is only a sort input, not an output
/// column, so the rule must still use the index. This is the regression guard for
/// the over-eager fallback: the output-aware passthrough drives the rewrite from
/// the OUTER projection (`[id]`, all producible), not the Sort's schema.
#[tokio::test]
async fn test_bare_select_inline_distance_still_rewrites() {
    let ctx = make_ctx().await;
    let sql = format!("SELECT id FROM items ORDER BY l2_distance(embedding, {Q}) ASC LIMIT 2");
    let plan = ctx
        .sql(&sql)
        .await
        .expect("SQL analysis failed")
        .into_optimized_plan()
        .expect("optimization must not error");
    assert!(
        contains_usearch_node(&plan),
        "vector not in output → rule must still use the index, not fall back\nPlan: {plan:?}"
    );
}
