// rule.rs — USearchRule: optimizer rewrite rule.
//
// Patterns matched (TopDown, Sort node):
//
//   Sort(fetch=k)
//     Projection([..., l2_distance(vector, lit) AS dist, ...])
//       TableScan(name)                        ← no WHERE clause
//
//   Sort(fetch=k)
//     Projection([..., l2_distance(vector, lit) AS dist, ...])
//       Filter(predicate)                      ← WHERE clause absorbed
//         TableScan(name)
//
// When a Filter node is present its predicate is stored in USearchNode.filters.
// The physical planner then runs adaptive filtered search:
//   - high selectivity → usearch::Index::filtered_search (in-graph filtering)
//   - low selectivity  → pre-filter scan + exact brute-force over valid subset
//
// Replacement:
//
//   Sort(fetch=k)                                             ← kept (sort order)
//     Projection([col(a), col(b), col("_distance").alias("dist")])
//       USearchNode                                           ← executes ANN

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::Array;
use datafusion::common::tree_node::Transformed;
use datafusion::common::{DFSchema, Result, TableReference};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNode, col};
use datafusion::logical_expr::logical_plan::Projection;
use datafusion::optimizer::OptimizerConfig;

use usearch::MetricKind;

use crate::node::{DistanceType, USearchNode};
use crate::registry::USearchRegistry;

pub struct USearchRule {
    registry: Arc<USearchRegistry>,
}

impl USearchRule {
    pub fn new(registry: Arc<USearchRegistry>) -> Self {
        Self { registry }
    }

    fn try_match(&self, plan: &LogicalPlan) -> Option<LogicalPlan> {
        use datafusion::logical_expr::logical_plan::TableScan;

        // Require Sort with embedded fetch limit.
        let sort = match plan {
            LogicalPlan::Sort(s) => s,
            _ => return None,
        };
        let k = sort.fetch?;

        // Require Projection immediately below Sort.
        let (proj_exprs, after_proj) = match sort.input.as_ref() {
            LogicalPlan::Projection(p) => (p.expr.as_slice(), p.input.as_ref()),
            _ => return None,
        };

        // Accept TableScan directly, or Filter(TableScan) for WHERE clauses.
        // Deeper nesting (Filter→Filter→…) is not absorbed — the rule does
        // not fire and DataFusion falls back to exact execution.
        let (table_name, filters) = match after_proj {
            LogicalPlan::TableScan(TableScan { table_name, .. }) => {
                (table_name.table().to_string(), vec![])
            }
            LogicalPlan::Filter(f) => match f.input.as_ref() {
                LogicalPlan::TableScan(TableScan { table_name, .. }) => {
                    (table_name.table().to_string(), vec![f.predicate.clone()])
                }
                _ => return None,
            },
            _ => return None,
        };

        // Table must be registered for vector search.
        let registered = self.registry.get(&table_name)?;

        // Find the distance UDF in the sort expressions.
        // Require ASC ordering — all three distance UDFs (l2_distance,
        // cosine_distance, negative_dot_product) return lower-is-closer values.
        let mut match_result: Option<(String, String, Vec<f32>, Option<String>)> = None;
        for sort_expr in &sort.expr {
            if let Some((udf_name, vec_col, query_vec)) =
                find_distance_info(&sort_expr.expr, Some(proj_exprs))
            {
                // Reject DESC sorts — e.g. ORDER BY negative_dot_product DESC is
                // asking for the *least* similar vectors; do not rewrite.
                if !sort_expr.asc {
                    return None;
                }
                let alias = extract_alias_name(&sort_expr.expr, proj_exprs);
                match_result = Some((udf_name, vec_col, query_vec, alias));
                break;
            }
        }

        let (udf_name, vec_col, query_vec, dist_alias) = match_result?;
        let dist_type = match udf_name.as_str() {
            "l2_distance" => DistanceType::L2,
            "cosine_distance" => DistanceType::Cosine,
            "negative_dot_product" => DistanceType::NegativeDot,
            _ => return None,
        };

        // Guard: the SQL distance UDF must match the metric the index was built
        // with. Mismatch → return None so DataFusion falls back to exact scan.
        if !dist_type_matches_metric(&dist_type, registered.metric) {
            return None;
        }

        // Build USearchNode schema: base fields qualified with the table name +
        // unqualified _distance. Qualifiers must match the original plan's schema.
        let table_ref = TableReference::bare(table_name.clone());
        let qualified_fields: Vec<(Option<TableReference>, Arc<arrow_schema::Field>)> =
            registered.schema.fields().iter().map(|f| {
                if f.name() == "_distance" {
                    (None, f.clone())
                } else {
                    (Some(table_ref.clone()), f.clone())
                }
            }).collect();
        let vsn_df_schema = DFSchema::new_with_metadata(qualified_fields, HashMap::new()).ok()?;

        let node = USearchNode::new(
            table_name.clone(),
            vec_col,
            query_vec,
            k,
            dist_type,
            Arc::new(vsn_df_schema),
            filters,
        );

        let node_plan = Arc::new(LogicalPlan::Extension(Extension {
            node: Arc::new(node) as Arc<dyn UserDefinedLogicalNode>,
        }));

        // Build Projection over USearchNode matching the original output schema.
        let dist_alias_str = dist_alias.as_deref().unwrap_or("_distance");
        let new_proj_exprs = remap_projections(proj_exprs, dist_alias_str, &table_name);
        let new_proj = Projection::try_new(new_proj_exprs, node_plan).ok()?;

        // Keep the Sort node so DataFusion handles ordering by _distance / dist.
        // USearch returns results in arbitrary (internal) order when the underlying
        // data is fetched from the TableProvider.
        Some(LogicalPlan::Sort(datafusion::logical_expr::logical_plan::Sort {
            expr: sort.expr.clone(),
            input: Arc::new(LogicalPlan::Projection(new_proj)),
            fetch: sort.fetch,
        }))
    }
}

impl std::fmt::Debug for USearchRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "USearchRule")
    }
}

impl datafusion::optimizer::OptimizerRule for USearchRule {
    fn name(&self) -> &str { "usearch_rule" }

    fn apply_order(&self) -> Option<datafusion::optimizer::ApplyOrder> {
        Some(datafusion::optimizer::ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool { true }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if let Some(new_plan) = self.try_match(&plan) {
            return Ok(Transformed::yes(new_plan));
        }
        Ok(Transformed::no(plan))
    }
}

// ── Pattern matching helpers (shared with vector_search/rule.rs) ──────────────

fn find_distance_info(
    expr: &Expr,
    proj_exprs: Option<&[Expr]>,
) -> Option<(String, String, Vec<f32>)> {
    if let Some(info) = try_extract_distance(expr) {
        return Some(info);
    }
    if let Expr::Column(col_ref) = expr {
        if let Some(projs) = proj_exprs {
            for proj in projs {
                let (alias_name, inner_expr): (Option<&str>, &Expr) = match proj {
                    Expr::Alias(a) => (Some(a.name.as_str()), a.expr.as_ref()),
                    other => (None, other),
                };
                if alias_name.map_or(false, |n| n == col_ref.name.as_str()) {
                    if let Some(info) = try_extract_distance(inner_expr) {
                        return Some(info);
                    }
                }
            }
        }
    }
    None
}

fn extract_alias_name(sort_expr: &Expr, proj_exprs: &[Expr]) -> Option<String> {
    match sort_expr {
        Expr::Column(c) => Some(c.name.clone()),
        Expr::Alias(a) => Some(a.name.clone()),
        Expr::ScalarFunction(sf) if is_dist_udf_name(sf.func.name()) => {
            proj_exprs.iter().find_map(|e| match e {
                Expr::Alias(a) if is_distance_expr(&a.expr) => Some(a.name.clone()),
                _ => None,
            })
        }
        _ => None,
    }
}

fn is_dist_udf_name(name: &str) -> bool {
    matches!(name, "l2_distance" | "cosine_distance" | "negative_dot_product")
}

/// Returns true if the SQL distance function matches the metric the index was built with.
/// Mismatch means the index would search by a different metric than the SQL requests,
/// producing wrong candidates and wrong _distance values silently.
fn dist_type_matches_metric(dist_type: &DistanceType, metric: MetricKind) -> bool {
    match dist_type {
        DistanceType::L2 => metric == MetricKind::L2sq,
        DistanceType::Cosine => metric == MetricKind::Cos,
        DistanceType::NegativeDot => metric == MetricKind::IP,
    }
}

fn is_distance_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(sf) if is_dist_udf_name(sf.func.name()))
}

fn try_extract_distance(expr: &Expr) -> Option<(String, String, Vec<f32>)> {
    let inner = match expr {
        Expr::Alias(a) => a.expr.as_ref(),
        other => other,
    };
    let sf = match inner {
        Expr::ScalarFunction(sf) => sf,
        _ => return None,
    };
    let udf_name = sf.func.name().to_string();
    if !is_dist_udf_name(&udf_name) { return None; }
    if sf.args.len() < 2 { return None; }

    let vec_col = match &sf.args[0] {
        Expr::Column(c) => c.name.clone(),
        _ => return None,
    };
    let query_vec = extract_f32_vec_from_expr(&sf.args[1])?;
    Some((udf_name, vec_col, query_vec))
}

fn remap_projections(proj_exprs: &[Expr], dist_alias_name: &str, table_name: &str) -> Vec<Expr> {
    proj_exprs.iter().map(|e| remap_one(e, dist_alias_name, table_name)).collect()
}

fn remap_one(expr: &Expr, dist_alias_name: &str, table_name: &str) -> Expr {
    match expr {
        Expr::Alias(a) if a.name == dist_alias_name && is_distance_expr(&a.expr) => {
            col("_distance").alias(a.name.as_str())
        }
        Expr::Alias(a) if is_distance_expr(&a.expr) => {
            col("_distance").alias(a.name.as_str())
        }
        Expr::Alias(a) => match a.expr.as_ref() {
            Expr::Column(c) => {
                col(format!("{table_name}.{}", c.name).as_str()).alias(a.name.as_str())
            }
            _ => col(a.name.as_str()),
        },
        Expr::Column(c) => col(format!("{table_name}.{}", c.name).as_str()),
        Expr::ScalarFunction(sf) if is_dist_udf_name(sf.func.name()) => col("_distance"),
        other => other.clone(),
    }
}

fn extract_f32_vec_from_expr(expr: &Expr) -> Option<Vec<f32>> {
    use arrow_array::{Float32Array, Float64Array};
    use datafusion::scalar::ScalarValue;

    match expr {
        // DataFusion 51: Expr::Literal is (ScalarValue, Option<FieldMetadata>)
        Expr::Literal(sv, _) => match sv {
            ScalarValue::FixedSizeList(arr) => {
                if arr.is_empty() { return None; }
                let inner = arr.value(0);
                if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
                    return Some(f32a.values().to_vec());
                }
                if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
                    return Some(f64a.values().iter().map(|&v| v as f32).collect());
                }
                None
            }
            ScalarValue::List(arr) => {
                if arr.is_empty() { return None; }
                let inner = arr.value(0);
                if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
                    return Some(f32a.values().to_vec());
                }
                if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
                    return Some(f64a.values().iter().map(|&v| v as f32).collect());
                }
                None
            }
            _ => None,
        },
        Expr::ScalarFunction(sf)
            if sf.func.name() == "make_array" || sf.func.name() == "array" =>
        {
            let mut result = Vec::with_capacity(sf.args.len());
            for arg in &sf.args {
                match arg {
                    Expr::Literal(ScalarValue::Float64(Some(v)), _) => result.push(*v as f32),
                    Expr::Literal(ScalarValue::Float32(Some(v)), _) => result.push(*v),
                    Expr::Literal(ScalarValue::Int64(Some(v)), _) => result.push(*v as f32),
                    Expr::Literal(ScalarValue::Int32(Some(v)), _) => result.push(*v as f32),
                    _ => return None,
                }
            }
            Some(result)
        }
        _ => None,
    }
}
