// udtf.rs — USearchUDTF: explicit SQL table function interface.
//
// Usage:
//   SELECT key, _distance FROM vector_usearch('table', ARRAY[...], k)
//   SELECT key, _distance FROM vector_usearch('table', ARRAY[...], k, ef_search)
//
// Returns two columns only: `key: UInt64` and `_distance: Float32`.
// To get full row data, JOIN the result against the data table:
//
//   SELECT d.id, d.name, vs._distance
//   FROM vector_usearch('items', ARRAY[...], 10) vs
//   JOIN items d ON d.id = vs.key
//   ORDER BY vs._distance

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::scalar::ScalarValue;

use crate::registry::USearchRegistry;

// ── UDTF ─────────────────────────────────────────────────────────────────────

/// Table function:  vector_usearch(table_name, query_vec, k [, ef_search])
///
/// Returns `(key: UInt64, _distance: Float32)`. Join with your data table on
/// the key column to retrieve full rows.
pub struct USearchUDTF {
    registry: Arc<USearchRegistry>,
}

impl fmt::Debug for USearchProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "USearchProvider({})", self.table_name)
    }
}

impl fmt::Debug for USearchUDTF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "USearchUDTF")
    }
}

impl USearchUDTF {
    pub fn new(registry: Arc<USearchRegistry>) -> Self {
        Self { registry }
    }
}

impl TableFunctionImpl for USearchUDTF {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if exprs.len() < 3 {
            return Err(DataFusionError::Execution(
                "vector_usearch requires at least 3 args: (table, query_vec, k)".into(),
            ));
        }

        let table_name = extract_string_literal(&exprs[0])?;
        let query_vec = extract_f32_vec(&exprs[1])?;
        let k = extract_usize_literal(&exprs[2])?;

        // Optional ef_search — used as a hint for the search expansion width.
        // NOTE: changing ef_search on a shared Arc<Index> affects all concurrent
        // queries. For production use, maintain separate index instances per
        // query, or set ef_search at load time.
        let _ef_search: Option<usize> = if exprs.len() > 3 {
            Some(extract_usize_literal(&exprs[3])?)
        } else {
            None
        };

        let registered = self.registry.get(&table_name).ok_or_else(|| {
            DataFusionError::Execution(format!(
                "vector_usearch: table '{table_name}' not registered"
            ))
        })?;

        Ok(Arc::new(USearchProvider {
            index: registered.index.clone(),
            table_name,
            query_vec,
            k,
        }))
    }
}

// ── TableProvider ─────────────────────────────────────────────────────────────

/// TableProvider returned by the UDTF. Executes a USearch ANN query in scan(),
/// returning only `(key: UInt64, _distance: Float32)`.
struct USearchProvider {
    index: Arc<usearch::Index>,
    table_name: String,
    query_vec: Vec<f32>,
    k: usize,
}

fn udtf_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt64, false),
        Field::new("_distance", DataType::Float32, true),
    ]))
}

#[async_trait]
impl TableProvider for USearchProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        udtf_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        let matches = self
            .index
            .search(&self.query_vec, self.k)
            .map_err(|e| DataFusionError::Execution(format!("USearch search error: {e}")))?;

        let schema = udtf_schema();

        let keys = UInt64Array::from(matches.keys.clone());
        let dists = Float32Array::from(matches.distances.clone());

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(keys), Arc::new(dists)])
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

        // Apply column projection so DataFusion's JOIN column indices are correct.
        let (proj_schema, proj_batches) = if let Some(indices) = projection {
            let ps = Arc::new(
                schema
                    .project(indices)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
            );
            let pb = batch
                .project(indices)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            (ps, vec![pb])
        } else {
            (schema, vec![batch])
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

// ── Literal extraction helpers ────────────────────────────────────────────────

// DataFusion 51: Expr::Literal is a 2-tuple (ScalarValue, Option<FieldMetadata>).

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
