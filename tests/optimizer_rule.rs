// tests/usearch_optimizer_rule.rs — Tests for USearchRule metric mismatch guard
// and sort-direction check.
//
// Setup: a synthetic "items" table (id: UInt64, vector: FixedSizeList<f32, 4>)
// is registered in USearchRegistry with a specific MetricKind and in the
// DataFusion SessionContext via HashKeyProvider (which also implements
// TableProvider, so no separate MemTable is needed).
//
// Tests only call into_optimized_plan() — physical planning never runs, so we
// don't need a USearchQueryPlanner or a populated index.
//
// Assertions: USearchNode present in the optimized plan ↔ rule fired.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};


use datafusion_vector_search_ext::{
    HashKeyProvider, USearchNode, USearchRegistry, register_all,
};

// ── Schema ────────────────────────────────────────────────────────────────────

fn items_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                4,
            ),
            false,
        ),
    ]))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_index(metric: MetricKind) -> Arc<Index> {
    let options = IndexOptions {
        dimensions: 4,
        metric,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    Arc::new(Index::new(&options).expect("usearch Index::new failed"))
}

/// Build a SessionContext with a USearch index registered under "items" using
/// the given MetricKind. Uses a plain SessionContext — no USearchQueryPlanner
/// needed because we only inspect the optimized logical plan (never .collect()).
async fn make_ctx(metric: MetricKind) -> SessionContext {
    let schema = items_schema();

    // Empty provider — no rows needed; we only test the optimizer rule.
    let provider = Arc::new(
        HashKeyProvider::try_new(schema.clone(), vec![], "id")
            .expect("HashKeyProvider::try_new failed"),
    );

    let reg = USearchRegistry::new();
    reg.add("items::vector", make_index(metric), provider.clone(), "id", metric, ScalarKind::F32)
        .expect("USearchRegistry::add failed");
    let registry = reg.into_arc();

    let ctx = SessionContext::default();
    register_all(&ctx, registry).expect("register_all failed");

    // HashKeyProvider also implements TableProvider, so register it directly
    // for SQL column-name resolution — no separate MemTable needed.
    ctx.register_table("items", provider).expect("register_table failed");

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
    if let LogicalPlan::Extension(ext) = plan {
        if ext.node.as_any().downcast_ref::<USearchNode>().is_some() {
            return true;
        }
    }
    plan.inputs().iter().any(|child| contains_usearch_node(child))
}

const Q: &str = "ARRAY[0.1, 0.2, 0.3, 0.4]";

// ═══════════════════════════════════════════════════════════════════════════════
// MATCHING METRIC — rule must fire
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_l2sq_index_l2_udf_rewrites() {
    let ctx = make_ctx(MetricKind::L2sq).await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        contains_usearch_node(&plan),
        "L2sq index + l2_distance ASC → rule must fire"
    );
}

#[tokio::test]
async fn test_cos_index_cosine_udf_rewrites() {
    let ctx = make_ctx(MetricKind::Cos).await;
    let sql = format!(
        "SELECT id, cosine_distance(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        contains_usearch_node(&plan),
        "Cos index + cosine_distance ASC → rule must fire"
    );
}

#[tokio::test]
async fn test_ip_index_negative_dot_udf_rewrites() {
    let ctx = make_ctx(MetricKind::IP).await;
    let sql = format!(
        "SELECT id, negative_dot_product(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        contains_usearch_node(&plan),
        "IP index + negative_dot_product ASC → rule must fire"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MISMATCHING METRIC — rule must NOT fire; query falls back to exact scan
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_l2sq_index_cosine_udf_no_rewrite() {
    let ctx = make_ctx(MetricKind::L2sq).await;
    let sql = format!(
        "SELECT id, cosine_distance(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "L2sq index + cosine_distance → metric mismatch, rule must not fire"
    );
}

#[tokio::test]
async fn test_l2sq_index_negative_dot_udf_no_rewrite() {
    let ctx = make_ctx(MetricKind::L2sq).await;
    let sql = format!(
        "SELECT id, negative_dot_product(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "L2sq index + negative_dot_product → metric mismatch, rule must not fire"
    );
}

#[tokio::test]
async fn test_cos_index_l2_udf_no_rewrite() {
    let ctx = make_ctx(MetricKind::Cos).await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "Cos index + l2_distance → metric mismatch, rule must not fire"
    );
}

#[tokio::test]
async fn test_cos_index_negative_dot_udf_no_rewrite() {
    let ctx = make_ctx(MetricKind::Cos).await;
    let sql = format!(
        "SELECT id, negative_dot_product(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "Cos index + negative_dot_product → metric mismatch, rule must not fire"
    );
}

#[tokio::test]
async fn test_ip_index_l2_udf_no_rewrite() {
    let ctx = make_ctx(MetricKind::IP).await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "IP index + l2_distance → metric mismatch, rule must not fire"
    );
}

#[tokio::test]
async fn test_ip_index_cosine_udf_no_rewrite() {
    let ctx = make_ctx(MetricKind::IP).await;
    let sql = format!(
        "SELECT id, cosine_distance(vector, {Q}) AS d FROM items ORDER BY d ASC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "IP index + cosine_distance → metric mismatch, rule must not fire"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESC SORT — rule must NOT fire even when metric matches
// ═══════════════════════════════════════════════════════════════════════════════

/// ORDER BY distance DESC asks for the *least* similar vectors — the ANN index
/// cannot answer that efficiently. The guard must reject it.
#[tokio::test]
async fn test_desc_sort_no_rewrite() {
    let ctx = make_ctx(MetricKind::L2sq).await;
    let sql = format!(
        "SELECT id, l2_distance(vector, {Q}) AS d FROM items ORDER BY d DESC LIMIT 5"
    );
    let plan = optimized_plan(&ctx, &sql).await;
    assert!(
        !contains_usearch_node(&plan),
        "l2_distance DESC (metric matches but wrong direction) → rule must not fire"
    );
}
