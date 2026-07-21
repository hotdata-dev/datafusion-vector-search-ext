// tests/subquery_alias.rs — Regression tests for table-aliasing SQL clients
// (ibis/SQLGlot, and any ORM/BI tool that aliases tables and wraps
// projections in subqueries).
//
// ibis emits SQL like:
//
//   SELECT * FROM (
//     SELECT "id", "label", COSINE_DISTANCE("vector", ARRAY[...]) AS "_distance"
//     FROM "items" AS "t0"
//   ) AS "t1" ORDER BY "t1"."_distance" ASC LIMIT 3
//
// which produces the logical plan:
//
//   Sort: t1._distance ASC, fetch=3
//     SubqueryAlias: t1
//       Projection: id, label, cosine_distance(...) AS _distance
//         SubqueryAlias: t0
//           TableScan: items
//
// DataFusion does not eliminate SubqueryAlias nodes, and USearchRule only
// descended through Projection/Filter — not SubqueryAlias — so it declined
// to fire and the query silently brute-forced.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use datafusion_vector_search_ext::{HashKeyProvider, USearchNode, USearchRegistry, register_all};

fn items_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("label", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
    ]))
}

fn make_index(metric: MetricKind) -> Arc<Index> {
    let options = IndexOptions {
        dimensions: 4,
        metric,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    Arc::new(Index::new(&options).expect("usearch Index::new failed"))
}

async fn make_ctx(metric: MetricKind) -> SessionContext {
    let schema = items_schema();
    let provider = Arc::new(
        HashKeyProvider::try_new(schema.clone(), vec![], "id")
            .expect("HashKeyProvider::try_new failed"),
    );

    let reg = USearchRegistry::new();
    reg.add(
        "items::vector",
        make_index(metric),
        provider.clone(),
        provider.clone(),
        "id",
        metric,
        ScalarKind::F32,
    )
    .expect("USearchRegistry::add failed");
    let registry = reg.into_arc();

    let ctx = SessionContext::default();
    register_all(&ctx, registry).expect("register_all failed");
    ctx.register_table("items", provider)
        .expect("register_table failed");
    ctx
}

async fn optimized_plan(ctx: &SessionContext, sql: &str) -> LogicalPlan {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|e| panic!("SQL parse/analysis failed: {e}\nSQL: {sql}"))
        .into_optimized_plan()
        .unwrap_or_else(|e| panic!("Optimization failed: {e}\nSQL: {sql}"))
}

fn contains_usearch_node(plan: &LogicalPlan) -> bool {
    if let LogicalPlan::Extension(ext) = plan
        && ext.node.as_any().downcast_ref::<USearchNode>().is_some()
    {
        return true;
    }
    plan.inputs()
        .iter()
        .any(|child| contains_usearch_node(child))
}

const Q: &str = "ARRAY[0.1, 0.2, 0.3, 0.4]";

// ═══════════════════════════════════════════════════════════════════════════════
// The ibis shape: outer SELECT * wraps an inner SELECT that aliases the scan.
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_double_alias_select_star_rewrites() {
    let ctx = make_ctx(MetricKind::Cos).await;
    let sql = format!(
        "SELECT * FROM (\
            SELECT id, label, cosine_distance(vector, {Q}) AS _distance \
            FROM items AS t0\
         ) AS t1 ORDER BY t1._distance ASC LIMIT 3"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        contains_usearch_node(&plan),
        "ibis-shape double SubqueryAlias → rule must fire\nPlan: {plan:#?}"
    );
}

/// Same shape but with an explicit outer column list instead of `SELECT *`.
#[tokio::test]
async fn test_double_alias_explicit_outer_projection_rewrites() {
    let ctx = make_ctx(MetricKind::Cos).await;
    let sql = format!(
        "SELECT id, _distance FROM (\
            SELECT id, label, cosine_distance(vector, {Q}) AS _distance \
            FROM items AS t0\
         ) AS t1 ORDER BY t1._distance ASC LIMIT 3"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        contains_usearch_node(&plan),
        "ibis-shape with explicit outer projection → rule must fire\nPlan: {plan:#?}"
    );
}

// Known limitation: a *single* alias directly on the scan, with no outer
// `SubqueryAlias` wrapping the whole `Sort`/output (unlike ibis, which always
// wraps its outer SELECT), leaves the Sort's own output schema with mixed
// qualifiers — e.g. `id` qualified `t0` (a plain passthrough of `t0.id`) but
// `dist` unqualified (an aliased/computed expression, DataFusion's own
// convention regardless of aliasing). DataFusion's `Projection` node can only
// reproduce a qualifier that already matches its input schema exactly (or
// none at all) — it can't stamp an arbitrary display qualifier onto one
// column. `SubqueryAlias` can, but only uniformly across every column it
// wraps, so re-wrapping in `t0` here would incorrectly requalify `dist` too.
// Reproducing this exact mixed schema would need per-column plan surgery
// disproportionate to a shape ibis doesn't actually emit (see module doc).
// The rule's safety net (`build_rewrite`'s trailing schema comparison)
// detects the mismatch and declines cleanly — no crash, just no rewrite,
// identical to pre-fix behavior for this shape.

/// Single alias level: no outer subquery, just `FROM items AS t0`.
#[tokio::test]
async fn test_single_alias_on_scan_no_outer_wrap_declines_safely() {
    let ctx = make_ctx(MetricKind::L2sq).await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS dist FROM items AS t0 ORDER BY dist ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "single aliased scan, no outer wrap → mixed-qualifier schema can't be reproduced; \
         rule must decline safely rather than fire\nPlan: {plan:#?}"
    );
}

/// Aliased scan with a WHERE clause — Filter(SubqueryAlias(TableScan)).
#[tokio::test]
async fn test_single_alias_with_where_no_outer_wrap_declines_safely() {
    let ctx = make_ctx(MetricKind::L2sq).await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS dist FROM items AS t0 \
         WHERE id > 5 ORDER BY dist ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "aliased scan + WHERE, no outer wrap → mixed-qualifier schema can't be reproduced; \
         rule must decline safely rather than fire\nPlan: {plan:#?}"
    );
}

/// Double alias + WHERE clause on the inner scan.
#[tokio::test]
async fn test_double_alias_with_where_rewrites() {
    let ctx = make_ctx(MetricKind::Cos).await;
    let sql = format!(
        "SELECT * FROM (\
            SELECT id, label, cosine_distance(vector, {Q}) AS _distance \
            FROM items AS t0 WHERE id > 0\
         ) AS t1 ORDER BY t1._distance ASC LIMIT 3"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        contains_usearch_node(&plan),
        "ibis-shape + WHERE → rule must fire\nPlan: {plan:#?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Vector column in output — must still decline even when aliased (issue #508).
// ═══════════════════════════════════════════════════════════════════════════════

/// Unlike `make_ctx`, the lookup provider here excludes the vector column —
/// mirroring production, where the fetch-path sidecar never stores it
/// (see tests/vector_col_projection.rs). Only this configuration can prove
/// the rule declines when the vector is unproducible; a lookup schema that
/// includes the vector (as `make_ctx`'s does) can't exercise that guard.
async fn make_ctx_no_vector_in_lookup(metric: MetricKind) -> SessionContext {
    let scan_provider = Arc::new(
        HashKeyProvider::try_new(items_schema(), vec![], "id")
            .expect("scan HashKeyProvider::try_new failed"),
    );
    let lookup_provider = Arc::new(
        HashKeyProvider::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("label", DataType::Utf8, true),
            ])),
            vec![],
            "id",
        )
        .expect("lookup HashKeyProvider::try_new failed"),
    );

    let reg = USearchRegistry::new();
    reg.add(
        "items::vector",
        make_index(metric),
        scan_provider,
        lookup_provider,
        "id",
        metric,
        ScalarKind::F32,
    )
    .expect("USearchRegistry::add failed");
    let registry = reg.into_arc();

    let ctx = SessionContext::default();
    register_all(&ctx, registry).expect("register_all failed");
    let table = Arc::new(
        HashKeyProvider::try_new(items_schema(), vec![], "id")
            .expect("table HashKeyProvider::try_new failed"),
    );
    ctx.register_table("items", table)
        .expect("register_table failed");
    ctx
}

#[tokio::test]
async fn test_double_alias_select_star_with_vector_no_rewrite() {
    let ctx = make_ctx_no_vector_in_lookup(MetricKind::Cos).await;
    let sql = format!(
        "SELECT * FROM (\
            SELECT * FROM items AS t0\
         ) AS t1 ORDER BY cosine_distance(t1.vector, {Q}) ASC LIMIT 3"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "aliased SELECT * (vector in output) → node can't produce vector, rule must not fire\nPlan: {plan:#?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Metric mismatch under aliasing — must still decline.
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_double_alias_metric_mismatch_no_rewrite() {
    let ctx = make_ctx(MetricKind::L2sq).await;
    let sql = format!(
        "SELECT * FROM (\
            SELECT id, label, cosine_distance(vector, {Q}) AS _distance \
            FROM items AS t0\
         ) AS t1 ORDER BY t1._distance ASC LIMIT 3"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "aliased shape + metric mismatch → rule must not fire\nPlan: {plan:#?}"
    );
}
