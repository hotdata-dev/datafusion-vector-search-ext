// node.rs — USearchNode: custom logical plan leaf node.
// The optimizer inserts this as a leaf when it rewrites a Sort/Limit/Scan
// pattern over a distance UDF into a direct USearch ANN call.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::{DFSchemaRef, Result};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Distance metric recognised from the SQL distance UDF.
/// Must match the MetricKind the USearch index was built with — the optimizer
/// rule validates this and will not fire on a mismatch.
/// All three variants use ORDER BY ASC (lower = more similar).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DistanceType {
    L2,
    Cosine,
    /// Negative inner product: -(A·B).  Matches MetricKind::IP (USearch stores
    /// 1 - A·B internally; both have the same ASC ordering).
    NegativeDot,
}

/// A logical plan leaf representing an ANN vector search against a registered
/// USearch index. Produced by USearchRule when it detects the canonical
/// Sort(Projection(TableScan)) + distance-UDF pattern, with or without an
/// intervening Filter node.
#[derive(Debug, Clone)]
pub struct USearchNode {
    pub table_name: String,
    pub vector_col: String,
    /// Query vector stored as bit-cast u32 to enable Hash/Eq/PartialOrd.
    query_vec_bits: Vec<u32>,
    pub k: usize,
    /// Distance metric matched from the SQL UDF. Validated against the index
    /// MetricKind before the optimizer rule fires.
    pub distance_type: DistanceType,
    pub schema: DFSchemaRef,
    /// Scalar filter predicates absorbed from the WHERE clause.
    /// Empty when no Filter node was present above the TableScan.
    /// The planner uses these to run adaptive filtered search:
    /// `filtered_search` at high selectivity, brute-force at low selectivity.
    pub filters: Vec<Expr>,
}

impl USearchNode {
    pub fn new(
        table_name: String,
        vector_col: String,
        query_vec: Vec<f32>,
        k: usize,
        distance_type: DistanceType,
        schema: DFSchemaRef,
        filters: Vec<Expr>,
    ) -> Self {
        let query_vec_bits = query_vec.iter().map(|v| v.to_bits()).collect();
        Self { table_name, vector_col, query_vec_bits, k, distance_type, schema, filters }
    }

    pub fn query_vec(&self) -> Vec<f32> {
        self.query_vec_bits.iter().map(|b| f32::from_bits(*b)).collect()
    }
}

impl PartialEq for USearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.table_name == other.table_name
            && self.vector_col == other.vector_col
            && self.query_vec_bits == other.query_vec_bits
            && self.k == other.k
            && self.distance_type == other.distance_type
            && self.filters == other.filters
    }
}

impl Eq for USearchNode {}

impl PartialOrd for USearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.table_name
            .partial_cmp(&other.table_name)
            .map(|o| o.then(self.k.cmp(&other.k)))
    }
}

impl Hash for USearchNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.table_name.hash(state);
        self.vector_col.hash(state);
        self.query_vec_bits.hash(state);
        self.k.hash(state);
        self.distance_type.hash(state);
        // Expr does not implement Hash; use its debug representation.
        // Different filter predicates must produce different hashes to prevent
        // incorrect common-subexpression elimination in the optimizer.
        format!("{:?}", self.filters).hash(state);
    }
}

impl UserDefinedLogicalNodeCore for USearchNode {
    fn name(&self) -> &str { "USearch" }

    fn inputs(&self) -> Vec<&LogicalPlan> { vec![] }

    fn schema(&self) -> &DFSchemaRef { &self.schema }

    fn expressions(&self) -> Vec<Expr> { vec![] }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "USearch: table={}, col={}, k={}, dist={:?}",
            self.table_name, self.vector_col, self.k, self.distance_type
        )?;
        if !self.filters.is_empty() {
            write!(f, ", filters={:?}", self.filters)?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, _inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(self.clone())
    }
}
