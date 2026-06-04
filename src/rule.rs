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
// Patterns matched (TopDown, Projection node):
//
//   Projection([output cols])                  ← SELECT list without the distance
//     Sort(l2_distance(vector, lit), fetch=k)  ← distance inline in ORDER BY
//       [Projection([output cols + vector])]   ← optional; DataFusion materializes
//         [Filter(predicate)]                     the vector only to feed the Sort
//           TableScan(name)
//
// In the Projection-anchored shape, producibility is judged on the OUTER
// projection (the query's real output): `SELECT id … ORDER BY l2_distance(…)`
// rewrites, while `SELECT *` / `SELECT id, vector` still fall back because the
// node cannot produce the vector column (issue #508).
//
// When a Filter node is present its predicate is stored in USearchNode.filters.
// The physical planner then runs adaptive filtered search:
//   - high selectivity → usearch::Index::filtered_search (in-graph filtering)
//   - low selectivity  → pre-filter scan + exact brute-force over valid subset
//
// Replacement:
//
//   Projection([final output cols])
//     Sort(fetch=k)
//       Projection([final output cols + optional hidden _distance])
//         USearchNode

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::Array;
use datafusion::common::tree_node::Transformed;
use datafusion::common::{DFSchema, Result, TableReference};
use datafusion::logical_expr::logical_plan::Projection;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNode, col};
use datafusion::optimizer::OptimizerConfig;

use usearch::MetricKind;

use crate::node::{DistanceType, USearchNode};
use crate::registry::VectorIndexResolver;

pub struct USearchRule {
    registry: Arc<dyn VectorIndexResolver>,
}

impl USearchRule {
    pub fn new(registry: Arc<dyn VectorIndexResolver>) -> Self {
        Self { registry }
    }

    fn try_match(&self, plan: &LogicalPlan) -> Option<LogicalPlan> {
        match plan {
            // Anchor on the Sort itself. The projection (if any) sits *below* the
            // Sort and supplies its output columns; SELECT * omits it entirely.
            LogicalPlan::Sort(sort) => {
                let (proj_exprs_slice, after_sort): (&[Expr], &LogicalPlan) =
                    match sort.input.as_ref() {
                        LogicalPlan::Projection(p) => (p.expr.as_slice(), p.input.as_ref()),
                        other => (&[], other),
                    };
                self.build_rewrite(sort, proj_exprs_slice, after_sort)
            }

            // Output-aware passthrough. When a Projection sits directly over a
            // k-NN Sort that rests on the scan (no projection between them), drive
            // the rewrite with the OUTER projection's columns — i.e. the query's
            // real output. The rewrite can only produce the index node's columns
            // (addressing key + non-vector columns + _distance), never the indexed
            // vector itself. Routing the output columns through `build_rewrite`
            // lets it fire when they're all producible (e.g. `SELECT id … ORDER BY
            // l2_distance(emb, …)`) and decline — falling back to exact search —
            // when the output needs the vector (`SELECT *`, `SELECT id, emb`),
            // rather than emitting a schema the consumer can't satisfy (issue #508).
            //
            // ALTERNATIVE (not taken): teach USearchExec to reconstruct the vector
            // column for the result keys via `index.get(key)`, so even
            // vector-returning queries stay on the index. Rejected to keep a single
            // source of truth for returned vectors — the index would otherwise be a
            // second source that must byte-match the parquet (breaks under F16
            // quantization, and relies on USearch never transforming stored vectors).
            // See the README "Limitations" entry and runtimedb issue #508.
            LogicalPlan::Projection(outer) => {
                let LogicalPlan::Sort(sort) = outer.input.as_ref() else {
                    return None;
                };
                match sort.input.as_ref() {
                    // Passthrough shape: Sort rests directly on the scan.
                    LogicalPlan::TableScan(_) | LogicalPlan::Filter(_) => {
                        self.build_rewrite(sort, &outer.expr, sort.input.as_ref())
                    }
                    // Trimmed shape: `SELECT id … ORDER BY l2_distance(vec, …)`
                    // with the distance NOT in the SELECT list. DataFusion
                    // materializes the raw vector column in an intermediate
                    // projection purely to feed the Sort, then trims it with
                    // this outer projection. Producibility must be judged on
                    // the OUTER (real output) columns — the inner projection's
                    // vector column never reaches the user. The Sort visit
                    // would wrongly decline this shape (it sees the vector
                    // among the inner projection's outputs and the node cannot
                    // produce it). When the distance is instead aliased inside
                    // the inner projection (`SELECT …, l2_distance(…) AS d …
                    // ORDER BY d`), `find_distance_info` finds no distance in
                    // the outer exprs, `build_rewrite` declines here, and the
                    // Sort visit handles it exactly as before.
                    LogicalPlan::Projection(inner)
                        if matches!(
                            inner.input.as_ref(),
                            LogicalPlan::TableScan(_) | LogicalPlan::Filter(_)
                        ) =>
                    {
                        self.build_rewrite(sort, &outer.expr, inner.input.as_ref())
                    }
                    _ => None,
                }
            }

            _ => None,
        }
    }

    fn build_rewrite(
        &self,
        sort: &datafusion::logical_expr::logical_plan::Sort,
        proj_exprs_slice: &[Expr],
        after_sort: &LogicalPlan,
    ) -> Option<LogicalPlan> {
        use datafusion::logical_expr::logical_plan::TableScan;

        // Require Sort with embedded fetch limit.
        let k = sort.fetch?;

        // Accept TableScan directly, or Filter(TableScan) for WHERE clauses.
        // Deeper nesting (Filter→Filter→…) is not absorbed — the rule does
        // not fire and DataFusion falls back to exact execution.
        let (table_ref_full, _table_name_bare, scan_table_ref, filters) = match after_sort {
            LogicalPlan::TableScan(TableScan { table_name, .. }) => (
                table_ref_to_str(table_name),
                table_name.table().to_string(),
                table_name.clone(),
                vec![],
            ),
            LogicalPlan::Filter(f) => match f.input.as_ref() {
                LogicalPlan::TableScan(TableScan { table_name, .. }) => (
                    table_ref_to_str(table_name),
                    table_name.table().to_string(),
                    table_name.clone(),
                    vec![f.predicate.clone()],
                ),
                _ => return None,
            },
            _ => return None,
        };

        // Find the distance UDF in the sort expressions first so we know the
        // vector column name before looking up the registry key.
        let mut pre_match: Option<(String, String, Vec<f64>, Option<String>)> = None;
        for sort_expr in &sort.expr {
            if let Some((udf_name, vec_col, query_vec)) =
                find_distance_info(&sort_expr.expr, Some(proj_exprs_slice))
            {
                if !sort_expr.asc {
                    return None;
                }
                let alias = extract_alias_name(&sort_expr.expr, proj_exprs_slice);
                pre_match = Some((udf_name, vec_col, query_vec, alias));
                break;
            }
        }
        let (udf_name, vec_col, query_vec, dist_alias) = pre_match?;

        // Registry key: "catalog::schema::table::col" (or fewer parts for bare refs).
        let reg_key = format!("{}::{}", table_ref_full, vec_col);

        // Sync check: does a vector index exist for this key?
        let meta = self.registry.peek(&reg_key)?;

        let dist_type = match udf_name.as_str() {
            "l2_distance" => DistanceType::L2,
            "cosine_distance" => DistanceType::Cosine,
            "negative_dot_product" => DistanceType::NegativeDot,
            _ => return None,
        };

        // Guard: the SQL distance UDF must match the metric the index was built
        // with. Mismatch → return None so DataFusion falls back to exact scan.
        if !dist_type_matches_metric(&dist_type, meta.metric) {
            return None;
        }

        // Build USearchNode schema: base fields qualified with the original table
        // reference (Full/Partial/Bare) so qualifiers match the original plan's schema.
        let table_ref = scan_table_ref.clone();
        let qualified_fields: Vec<(Option<TableReference>, Arc<arrow_schema::Field>)> = meta
            .schema
            .fields()
            .iter()
            .map(|f| {
                if f.name() == "_distance" {
                    (None, f.clone())
                } else {
                    (Some(table_ref.clone()), f.clone())
                }
            })
            .collect();
        let vsn_df_schema = DFSchema::new_with_metadata(qualified_fields, HashMap::new()).ok()?;

        let node = USearchNode::new(
            reg_key.clone(),
            vec_col,
            query_vec,
            k,
            dist_type,
            Arc::new(vsn_df_schema.clone()),
            filters,
        );

        let node_plan = Arc::new(LogicalPlan::Extension(Extension {
            node: Arc::new(node) as Arc<dyn UserDefinedLogicalNode>,
        }));

        // Build the final user-visible projection over USearchNode output.
        let dist_alias_str = dist_alias.as_deref().unwrap_or("_distance");
        let final_proj_exprs = if proj_exprs_slice.is_empty() {
            // No explicit Projection node (e.g. SELECT *, or a SELECT whose
            // columns come straight from the scan, so the Sort sits directly on
            // the TableScan). The rewrite must reproduce the original output
            // columns; if any isn't producible from the node — the indexed
            // vector column is never stored in the fetch path — bail so the
            // query falls back to exact brute-force search, like the other
            // unsupported shapes (DESC, metric mismatch, stacked filters).
            passthrough_projection(after_sort.schema().as_ref(), &vsn_df_schema, &table_ref)?
        } else {
            remap_projections(proj_exprs_slice, dist_alias_str, &table_ref)
        };
        let remapped_sort_exprs = remap_sort_exprs(&sort.expr, dist_alias.as_deref());
        let needs_hidden_distance = remapped_sort_exprs.iter().any(
            |e| matches!(&e.expr, Expr::Column(c) if c.relation.is_none() && c.name == "_distance"),
        ) && !projection_exposes_name(&final_proj_exprs, "_distance");

        let mut sort_input_exprs = final_proj_exprs.clone();
        if needs_hidden_distance {
            sort_input_exprs.push(col("_distance"));
        }

        let sort_input = Projection::try_new(sort_input_exprs, node_plan).ok()?;
        let sorted = LogicalPlan::Sort(datafusion::logical_expr::logical_plan::Sort {
            expr: remapped_sort_exprs,
            input: Arc::new(LogicalPlan::Projection(sort_input)),
            fetch: sort.fetch,
        });

        let outer_proj_exprs = build_outer_projection(&final_proj_exprs);
        let outer_proj = Projection::try_new(outer_proj_exprs, Arc::new(sorted)).ok()?;
        Some(LogicalPlan::Projection(outer_proj))
    }
}

impl std::fmt::Debug for USearchRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "USearchRule")
    }
}

impl datafusion::optimizer::OptimizerRule for USearchRule {
    fn name(&self) -> &str {
        "usearch_rule"
    }

    fn apply_order(&self) -> Option<datafusion::optimizer::ApplyOrder> {
        Some(datafusion::optimizer::ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

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

// ── Table reference helpers ───────────────────────────────────────────────────

/// Convert a [`TableReference`] to a `"::"` separated string.
///
/// Used as the prefix for registry keys: `"catalog::schema::table::col"`.
fn table_ref_to_str(r: &TableReference) -> String {
    match r {
        TableReference::Full {
            catalog,
            schema,
            table,
        } => {
            format!("{}::{}::{}", catalog, schema, table)
        }
        TableReference::Partial { schema, table } => format!("{}::{}", schema, table),
        TableReference::Bare { table } => table.to_string(),
    }
}

// ── Pattern matching helpers (shared with vector_search/rule.rs) ──────────────

fn find_distance_info(
    expr: &Expr,
    proj_exprs: Option<&[Expr]>,
) -> Option<(String, String, Vec<f64>)> {
    if let Some(info) = try_extract_distance(expr) {
        return Some(info);
    }
    if let Expr::Column(col_ref) = expr
        && let Some(projs) = proj_exprs
    {
        for proj in projs {
            let (alias_name, inner_expr): (Option<&str>, &Expr) = match proj {
                Expr::Alias(a) => (Some(a.name.as_str()), a.expr.as_ref()),
                other => (None, other),
            };
            if alias_name == Some(col_ref.name.as_str())
                && let Some(info) = try_extract_distance(inner_expr)
            {
                return Some(info);
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
    matches!(
        name,
        "l2_distance" | "cosine_distance" | "negative_dot_product"
    )
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
    let inner = match expr {
        Expr::Alias(a) => a.expr.as_ref(),
        other => other,
    };
    matches!(inner, Expr::ScalarFunction(sf) if is_dist_udf_name(sf.func.name()))
}

fn try_extract_distance(expr: &Expr) -> Option<(String, String, Vec<f64>)> {
    let inner = match expr {
        Expr::Alias(a) => a.expr.as_ref(),
        other => other,
    };
    let sf = match inner {
        Expr::ScalarFunction(sf) => sf,
        _ => return None,
    };
    let udf_name = sf.func.name().to_string();
    if !is_dist_udf_name(&udf_name) {
        return None;
    }
    if sf.args.len() < 2 {
        return None;
    }

    let vec_col = match &sf.args[0] {
        Expr::Column(c) => c.name.clone(),
        _ => return None,
    };
    let query_vec = extract_f64_vec_from_expr(&sf.args[1])?;
    Some((udf_name, vec_col, query_vec))
}

fn remap_projections(
    proj_exprs: &[Expr],
    dist_alias_name: &str,
    table_ref: &TableReference,
) -> Vec<Expr> {
    proj_exprs
        .iter()
        .map(|e| remap_one(e, dist_alias_name, table_ref))
        .collect()
}

fn remap_sort_exprs(
    sort_exprs: &[datafusion::logical_expr::SortExpr],
    dist_alias_name: Option<&str>,
) -> Vec<datafusion::logical_expr::SortExpr> {
    sort_exprs
        .iter()
        .map(|sort_expr| {
            let remapped_expr = match &sort_expr.expr {
                Expr::Column(c) if Some(c.name.as_str()) == dist_alias_name => col(c.name.as_str()),
                expr if is_distance_expr(expr) => col("_distance"),
                other => other.clone(),
            };
            datafusion::logical_expr::SortExpr {
                expr: remapped_expr,
                asc: sort_expr.asc,
                nulls_first: sort_expr.nulls_first,
            }
        })
        .collect()
}

fn projection_exposes_name(exprs: &[Expr], name: &str) -> bool {
    exprs.iter().any(|expr| match expr {
        Expr::Alias(a) => a.name == name,
        Expr::Column(c) => c.name == name,
        _ => false,
    })
}

fn build_outer_projection(exprs: &[Expr]) -> Vec<Expr> {
    exprs
        .iter()
        .map(|expr| match expr {
            Expr::Alias(a) => col(a.name.as_str()),
            Expr::Column(c) => Expr::Column(c.clone()),
            other => col(other.schema_name().to_string()),
        })
        .collect()
}

/// Build a passthrough Projection for queries with no explicit Projection node
/// (e.g. `SELECT *`, or a SELECT whose columns come straight from the scan so the
/// Sort sits directly on the TableScan).
///
/// The projection must reproduce the *original* output columns (`original_schema`,
/// the Sort's input). The `USearchNode` can only produce the columns in
/// `node_schema` — the fetch path's addressing key + non-vector columns +
/// `_distance`; the indexed vector column is never stored there (see
/// `PointLookupProvider`). If the original output needs a column the node can't
/// produce (the vector column), return `None` so the rule declines to rewrite and
/// the query falls back to exact brute-force search. The Sort re-evaluates the
/// distance UDF on the k result rows returned by USearchExec (O(k × dim)).
fn passthrough_projection(
    original_schema: &DFSchema,
    node_schema: &DFSchema,
    table_ref: &TableReference,
) -> Option<Vec<Expr>> {
    original_schema
        .inner()
        .fields()
        .iter()
        .map(|f| {
            let producible = node_schema
                .inner()
                .fields()
                .iter()
                .any(|nf| nf.name() == f.name());
            producible.then(|| {
                Expr::Column(datafusion::common::Column::new(
                    Some(table_ref.clone()),
                    f.name().as_str(),
                ))
            })
        })
        .collect()
}

fn remap_one(expr: &Expr, dist_alias_name: &str, table_ref: &TableReference) -> Expr {
    match expr {
        Expr::Alias(a) if a.name == dist_alias_name && is_distance_expr(&a.expr) => {
            col("_distance").alias(a.name.as_str())
        }
        Expr::Alias(a) if is_distance_expr(&a.expr) => col("_distance").alias(a.name.as_str()),
        Expr::Alias(a) => match a.expr.as_ref() {
            Expr::Column(c) => Expr::Column(datafusion::common::Column::new(
                Some(table_ref.clone()),
                c.name.as_str(),
            ))
            .alias(a.name.as_str()),
            _ => col(a.name.as_str()),
        },
        Expr::Column(c) => Expr::Column(datafusion::common::Column::new(
            Some(table_ref.clone()),
            c.name.as_str(),
        )),
        Expr::ScalarFunction(sf) if is_dist_udf_name(sf.func.name()) => col("_distance"),
        other => other.clone(),
    }
}

/// Extract a query vector as `Vec<f64>` from a SQL literal expression.
///
/// Preserves full f64 precision so that SQL literals like `ARRAY[0.123456789012345]`
/// are not silently rounded to f32 before reaching the index search.  The
/// planner casts to the index's native scalar kind (f32/f64) at the last moment.
fn extract_f64_vec_from_expr(expr: &Expr) -> Option<Vec<f64>> {
    use arrow_array::{Float32Array, Float64Array};
    use datafusion::scalar::ScalarValue;

    match expr {
        // DataFusion 51: Expr::Literal is (ScalarValue, Option<FieldMetadata>)
        Expr::Literal(sv, _) => match sv {
            ScalarValue::FixedSizeList(arr) => {
                if arr.is_empty() {
                    return None;
                }
                let inner = arr.value(0);
                if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
                    return Some(f64a.values().to_vec());
                }
                if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
                    return Some(f32a.values().iter().map(|&v| v as f64).collect());
                }
                None
            }
            ScalarValue::List(arr) => {
                if arr.is_empty() {
                    return None;
                }
                let inner = arr.value(0);
                if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
                    return Some(f64a.values().to_vec());
                }
                if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
                    return Some(f32a.values().iter().map(|&v| v as f64).collect());
                }
                None
            }
            _ => None,
        },
        Expr::ScalarFunction(sf) if sf.func.name() == "make_array" || sf.func.name() == "array" => {
            let mut result = Vec::with_capacity(sf.args.len());
            for arg in &sf.args {
                match arg {
                    Expr::Literal(ScalarValue::Float64(Some(v)), _) => result.push(*v),
                    Expr::Literal(ScalarValue::Float32(Some(v)), _) => result.push(*v as f64),
                    Expr::Literal(ScalarValue::Int64(Some(v)), _) => result.push(*v as f64),
                    Expr::Literal(ScalarValue::Int32(Some(v)), _) => result.push(*v as f64),
                    _ => return None,
                }
            }
            Some(result)
        }
        _ => None,
    }
}
