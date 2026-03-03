// udf.rs — scalar UDFs for distance computation.
// Three distance functions: l2_distance, cosine_distance, dot_product.
// Each takes (vector_col: FixedSizeList<Float32>, query: Array/Scalar) and
// returns a Float32 distance per row.
//
// These are identical to the vector_search UDFs but kept in this module so
// vector_usearch is fully self-contained (no dependency on vector_search).

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use datafusion::scalar::ScalarValue;

type Kernel = fn(&[f32], &[f32]) -> f32;

fn l2_kernel(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

fn cosine_kernel(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

fn negative_dot_kernel(a: &[f32], b: &[f32]) -> f32 {
    -(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>())
}

pub struct DistanceUDF {
    name: String,
    signature: Signature,
    kernel: Kernel,
}

impl DistanceUDF {
    fn new(name: &str, kernel: Kernel) -> Self {
        let signature = Signature::new(TypeSignature::VariadicAny, Volatility::Immutable);
        Self { name: name.to_string(), signature, kernel }
    }
}

impl std::fmt::Debug for DistanceUDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DistanceUDF({})", self.name)
    }
}

impl PartialEq for DistanceUDF {
    fn eq(&self, other: &Self) -> bool { self.name == other.name }
}
impl Eq for DistanceUDF {}
impl Hash for DistanceUDF {
    fn hash<H: Hasher>(&self, state: &mut H) { self.name.hash(state); }
}

impl ScalarUDFImpl for DistanceUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { &self.name }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let cols = &args.args;
        if cols.len() < 2 {
            return Err(DataFusionError::Execution(format!(
                "{}: expected 2 arguments, got {}",
                self.name,
                cols.len()
            )));
        }

        let vec_col = match &cols[0] {
            ColumnarValue::Array(arr) => arr.clone(),
            ColumnarValue::Scalar(sv) => sv.to_array()?,
        };
        let query_vec: Vec<f32> = extract_query_vec(&cols[1])?;

        let fsl = vec_col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "{}: arg0 must be FixedSizeListArray, got {:?}",
                    self.name,
                    vec_col.data_type()
                ))
            })?;

        let kernel = self.kernel;
        let mut distances = Vec::with_capacity(fsl.len());
        for i in 0..fsl.len() {
            if fsl.is_null(i) {
                distances.push(None);
                continue;
            }
            let row = fsl.value(i);
            let row_f32 = row.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "{}: inner array must be Float32Array",
                    self.name
                ))
            })?;
            distances.push(Some(kernel(row_f32.values(), &query_vec)));
        }

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(distances))))
    }
}

fn extract_query_vec(val: &ColumnarValue) -> Result<Vec<f32>> {
    match val {
        ColumnarValue::Scalar(sv) => scalar_to_f32_vec(sv),
        ColumnarValue::Array(arr) => {
            if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
                if fsl.is_empty() {
                    return Err(DataFusionError::Execution("query vec is empty".into()));
                }
                let inner = fsl.value(0);
                let f32arr = inner.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                    DataFusionError::Execution("query vec inner must be Float32Array".into())
                })?;
                return Ok(f32arr.values().to_vec());
            }
            let f32arr = arr.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "Cannot interpret query arg as f32 vec: {:?}",
                    arr.data_type()
                ))
            })?;
            Ok(f32arr.values().to_vec())
        }
    }
}

fn scalar_to_f32_vec(sv: &ScalarValue) -> Result<Vec<f32>> {
    use arrow_array::{Float32Array, Float64Array};
    match sv {
        ScalarValue::FixedSizeList(arr) => {
            let inner = arr.value(0);
            if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
                return Ok(f32a.values().to_vec());
            }
            if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
                return Ok(f64a.values().iter().map(|&v| v as f32).collect());
            }
            Err(DataFusionError::Execution("FixedSizeList inner is not Float32/Float64".into()))
        }
        ScalarValue::List(arr) => {
            let inner = arr.value(0);
            if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
                return Ok(f32a.values().to_vec());
            }
            if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
                return Ok(f64a.values().iter().map(|&v| v as f32).collect());
            }
            Err(DataFusionError::Execution("List scalar inner is not Float32/Float64".into()))
        }
        other => Err(DataFusionError::Execution(format!(
            "Unsupported scalar type for query vec: {:?}",
            other
        ))),
    }
}

pub fn l2_distance_udf() -> DistanceUDF { DistanceUDF::new("l2_distance", l2_kernel) }
pub fn cosine_distance_udf() -> DistanceUDF { DistanceUDF::new("cosine_distance", cosine_kernel) }
pub fn negative_dot_product_udf() -> DistanceUDF { DistanceUDF::new("negative_dot_product", negative_dot_kernel) }
