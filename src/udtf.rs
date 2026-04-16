// udtf.rs — vector_search_vector: explicit SQL table function for ANN search.
//
// Usage:
//   SELECT * FROM vector_search_vector('conn.schema.table', 'column', ARRAY[...], k)
//
// Returns all table columns plus `_distance: Float32`.
// Requires a vector index on the specified column.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::scalar::ScalarValue;

use crate::planner::{attach_distances, provider_key_col_idx, usearch_search};
use crate::registry::{RegisteredTable, VectorIndexResolver};

// ── UDTF ─────────────────────────────────────────────────────────────────────

/// Table function: vector_search_vector('conn.schema.table', 'column', ARRAY[...], k)
///
/// Returns all table columns plus `_distance: Float32`.
///
/// This entry point is synchronous. It calls `resolve()` (cache-only) on the
/// registry, so the index must already be loaded (e.g. via `refresh_for_tables`
/// before planning). If the index is not cached, the function returns an error.
pub struct VectorSearchVectorUDTF {
    registry: Arc<dyn VectorIndexResolver>,
}

impl fmt::Debug for VectorSearchVectorUDTF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorSearchVectorUDTF")
    }
}

impl VectorSearchVectorUDTF {
    pub fn new(registry: Arc<dyn VectorIndexResolver>) -> Self {
        Self { registry }
    }
}

impl TableFunctionImpl for VectorSearchVectorUDTF {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if exprs.len() != 4 {
            return Err(DataFusionError::Plan(
                "vector_search_vector requires 4 arguments: \
                 vector_search_vector('conn.schema.table', 'column', ARRAY[...], k)"
                    .into(),
            ));
        }

        let table_ref = extract_string_literal(&exprs[0])?;
        let column = extract_string_literal(&exprs[1])?;
        let query_vec = extract_f32_vec(&exprs[2])?;
        let k = extract_usize_literal(&exprs[3])?;

        // Build the registry key: "conn::schema::table::column"
        let (conn, schema, table) = parse_dot_table_ref(&table_ref)?;
        let reg_key = format!("{conn}::{schema}::{table}::{column}");

        let registered = self.registry.resolve(&reg_key).ok_or_else(|| {
            DataFusionError::Execution(format!(
                "vector_search_vector: no loaded vector index for '{reg_key}'. \
                 Ensure the table is synced and has a vector index on column '{column}'."
            ))
        })?;

        Ok(Arc::new(VectorSearchVectorProvider {
            registered,
            query_vec,
            k,
        }))
    }
}

// ── TableProvider ─────────────────────────────────────────────────────────────

/// TableProvider returned by the UDTF. Executes a USearch ANN query in scan(),
/// fetches full rows via the lookup provider, and appends `_distance`.
struct VectorSearchVectorProvider {
    registered: Arc<RegisteredTable>,
    query_vec: Vec<f32>,
    k: usize,
}

impl fmt::Debug for VectorSearchVectorProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorSearchVectorProvider(k={})", self.k)
    }
}

#[async_trait]
impl TableProvider for VectorSearchVectorProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        // RegisteredTable.schema already includes all data columns + _distance
        self.registered.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // 1. HNSW search
        let query_f64: Vec<f64> = self.query_vec.iter().map(|&v| v as f64).collect();
        let matches = usearch_search(
            &self.registered.index,
            &query_f64,
            self.k,
            self.registered.scalar_kind,
        )?;

        if matches.keys.is_empty() {
            let schema = match projection {
                Some(indices) => Arc::new(
                    self.registered
                        .schema
                        .project(indices)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
                ),
                None => self.registered.schema.clone(),
            };
            return Ok(Arc::new(BatchExec::new(schema, vec![])));
        }

        // 2. Build key → distance map
        let key_to_dist: HashMap<u64, f32> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .map(|(&k, &d)| (k, d))
            .collect();

        // 3. Fetch full rows from lookup provider
        let data_batches = self
            .registered
            .lookup_provider
            .fetch_by_keys(&matches.keys, &self.registered.key_col, None)
            .await?;

        // 4. Attach _distance column
        let key_col_idx = provider_key_col_idx(&self.registered)?;
        let result_batches = attach_distances(
            data_batches,
            key_col_idx,
            &key_to_dist,
            &self.registered.schema,
        )?;

        // 5. Apply projection if needed
        let (proj_schema, proj_batches) = if let Some(indices) = projection {
            let ps = Arc::new(
                self.registered
                    .schema
                    .project(indices)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
            );
            let pb: Vec<RecordBatch> = result_batches
                .into_iter()
                .map(|b| {
                    b.project(indices)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                })
                .collect::<Result<_>>()?;
            (ps, pb)
        } else {
            (self.registered.schema.clone(), result_batches)
        };

        Ok(Arc::new(BatchExec::new(proj_schema, proj_batches)))
    }
}

// ── Simple pre-computed batch execution plan ──────────────────────────────────

#[derive(Debug)]
struct BatchExec {
    schema: SchemaRef,
    batches: Arc<Vec<RecordBatch>>,
    properties: PlanProperties,
}

impl BatchExec {
    fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            schema,
            batches: Arc::new(batches),
            properties,
        }
    }
}

impl DisplayAs for BatchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BatchExec: {} batches", self.batches.len())
    }
}

impl ExecutionPlan for BatchExec {
    fn name(&self) -> &str {
        "BatchExec"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "BatchExec is a leaf node".to_string(),
            ))
        }
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(MemoryStream::try_new(
            self.batches.as_ref().clone(),
            self.schema.clone(),
            None,
        )?))
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Parse a dot-separated table reference: "conn.schema.table" → ("conn", "schema", "table")
fn parse_dot_table_ref(s: &str) -> Result<(String, String, String)> {
    let parts: Vec<&str> = s.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(DataFusionError::Plan(format!(
            "Expected 'connection.schema.table', got '{s}'"
        )));
    }
    Ok((
        parts[0].to_string(),
        parts[1].to_string(),
        parts[2].to_string(),
    ))
}

// DataFusion 51+: Expr::Literal is a 2-tuple (ScalarValue, Option<FieldMetadata>).

fn extract_string_literal(expr: &Expr) -> Result<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _) => Ok(s.clone()),
        Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => Ok(s.clone()),
        other => Err(DataFusionError::Execution(format!(
            "Expected string literal, got: {other:?}"
        ))),
    }
}

fn extract_usize_literal(expr: &Expr) -> Result<usize> {
    match expr {
        Expr::Literal(ScalarValue::Int64(Some(v)), _) => Ok(*v as usize),
        Expr::Literal(ScalarValue::Int32(Some(v)), _) => Ok(*v as usize),
        Expr::Literal(ScalarValue::UInt64(Some(v)), _) => Ok(*v as usize),
        Expr::Literal(ScalarValue::UInt32(Some(v)), _) => Ok(*v as usize),
        other => Err(DataFusionError::Execution(format!(
            "Expected integer literal, got: {other:?}"
        ))),
    }
}

fn extract_f32_vec(expr: &Expr) -> Result<Vec<f32>> {
    use arrow_array::{Float32Array, Float64Array};

    match expr {
        Expr::Literal(ScalarValue::FixedSizeList(arr), _) => {
            if arr.is_empty() {
                return Err(DataFusionError::Execution("Empty query vector".into()));
            }
            let inner = arr.value(0);
            if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
                return Ok(f32a.values().to_vec());
            }
            if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
                return Ok(f64a.values().iter().map(|&v| v as f32).collect());
            }
            Err(DataFusionError::Execution(
                "FixedSizeList inner is not Float32/Float64".into(),
            ))
        }
        Expr::Literal(ScalarValue::List(arr), _) => {
            if arr.is_empty() {
                return Err(DataFusionError::Execution("Empty query vector".into()));
            }
            let inner = arr.value(0);
            if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
                return Ok(f32a.values().to_vec());
            }
            if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
                return Ok(f64a.values().iter().map(|&v| v as f32).collect());
            }
            Err(DataFusionError::Execution(
                "List scalar inner is not Float32/Float64".into(),
            ))
        }
        Expr::ScalarFunction(sf) if sf.func.name() == "make_array" || sf.func.name() == "array" => {
            let mut result = Vec::with_capacity(sf.args.len());
            for arg in &sf.args {
                match arg {
                    Expr::Literal(ScalarValue::Float64(Some(v)), _) => result.push(*v as f32),
                    Expr::Literal(ScalarValue::Float32(Some(v)), _) => result.push(*v),
                    Expr::Literal(ScalarValue::Int64(Some(v)), _) => result.push(*v as f32),
                    Expr::Literal(ScalarValue::Int32(Some(v)), _) => result.push(*v as f32),
                    other => {
                        return Err(DataFusionError::Execution(format!(
                            "Non-literal in ARRAY[...]: {other:?}"
                        )));
                    }
                }
            }
            Ok(result)
        }
        other => Err(DataFusionError::Execution(format!(
            "Cannot extract f32 vector from: {other:?}"
        ))),
    }
}
