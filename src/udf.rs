// udf.rs — scalar UDFs for distance computation.
// Three distance functions: l2_distance, cosine_distance, dot_product.
// Each takes (vector_col: FixedSizeList<Float32>, query: Array/Scalar) and
// returns a Float32 distance per row.
//
// These are kept in this module alongside the UDTF and optimizer rule.

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
};
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use datafusion::scalar::ScalarValue;

type Kernel = fn(&[f32], &[f32]) -> f32;

// Returns L2sq (no sqrt) — matches USearch MetricKind::L2sq and keeps numeric
// values consistent between the UDF path and the optimizer-rewritten index path.
fn l2_kernel(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
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
        Self {
            name: name.to_string(),
            signature,
            kernel,
        }
    }
}

impl std::fmt::Debug for DistanceUDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DistanceUDF({})", self.name)
    }
}

impl PartialEq for DistanceUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for DistanceUDF {}
impl Hash for DistanceUDF {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl ScalarUDFImpl for DistanceUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
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

        let kernel = self.kernel;
        let name = &self.name;
        let distances = compute_distances(&vec_col, &query_vec, kernel, name)?;

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(
            distances,
        ))))
    }
}

/// Extract f32 values from a row's inner array, casting Float64 → f32 if needed.
fn inner_to_f32(inner: &dyn Array, udf_name: &str) -> Result<Vec<f32>> {
    if let Some(f32a) = inner.as_any().downcast_ref::<Float32Array>() {
        return Ok(f32a.values().to_vec());
    }
    if let Some(f64a) = inner.as_any().downcast_ref::<Float64Array>() {
        return Ok(f64a.values().iter().map(|&v| v as f32).collect());
    }
    Err(DataFusionError::Execution(format!(
        "{udf_name}: inner element type must be Float32 or Float64, got {:?}",
        inner.data_type()
    )))
}

/// Compute per-row distances from a vector column and a query slice.
///
/// Supports all outer array types (FixedSizeList, List, LargeList) and
/// inner element types (Float32, Float64 — Float64 is cast to f32 for the kernel).
///
/// Returns an error if vector dimensions do not match the query length.
fn compute_distances(
    vec_col: &dyn Array,
    query_vec: &[f32],
    kernel: Kernel,
    udf_name: &str,
) -> Result<Vec<Option<f32>>> {
    // FixedSizeListArray — dimension known from type, validate once upfront.
    if let Some(fsl) = vec_col.as_any().downcast_ref::<FixedSizeListArray>() {
        let col_dim = fsl.value_length() as usize;
        if col_dim != query_vec.len() {
            return Err(DataFusionError::Execution(format!(
                "{udf_name}: query vector length ({}) must match column dimensionality ({col_dim})",
                query_vec.len(),
            )));
        }
        let mut out = Vec::with_capacity(fsl.len());
        for i in 0..fsl.len() {
            if fsl.is_null(i) {
                out.push(None);
                continue;
            }
            let f32s = inner_to_f32(&*fsl.value(i), udf_name)?;
            out.push(Some(kernel(&f32s, query_vec)));
        }
        return Ok(out);
    }

    // ListArray — variable-length, validate per row.
    if let Some(lst) = vec_col.as_any().downcast_ref::<ListArray>() {
        let mut out = Vec::with_capacity(lst.len());
        for i in 0..lst.len() {
            if lst.is_null(i) {
                out.push(None);
                continue;
            }
            let f32s = inner_to_f32(&*lst.value(i), udf_name)?;
            if f32s.len() != query_vec.len() {
                return Err(DataFusionError::Execution(format!(
                    "{udf_name}: query vector length ({}) must match row {i} dimensionality ({})",
                    query_vec.len(),
                    f32s.len(),
                )));
            }
            out.push(Some(kernel(&f32s, query_vec)));
        }
        return Ok(out);
    }

    // LargeListArray — large-offset variant, validate per row.
    if let Some(lst) = vec_col.as_any().downcast_ref::<LargeListArray>() {
        let mut out = Vec::with_capacity(lst.len());
        for i in 0..lst.len() {
            if lst.is_null(i) {
                out.push(None);
                continue;
            }
            let f32s = inner_to_f32(&*lst.value(i), udf_name)?;
            if f32s.len() != query_vec.len() {
                return Err(DataFusionError::Execution(format!(
                    "{udf_name}: query vector length ({}) must match row {i} dimensionality ({})",
                    query_vec.len(),
                    f32s.len(),
                )));
            }
            out.push(Some(kernel(&f32s, query_vec)));
        }
        return Ok(out);
    }

    Err(DataFusionError::Execution(format!(
        "{udf_name}: arg0 must be FixedSizeList, List, or LargeList(Float32/Float64), got {:?}",
        vec_col.data_type()
    )))
}

fn extract_query_vec(val: &ColumnarValue) -> Result<Vec<f32>> {
    match val {
        ColumnarValue::Scalar(sv) => scalar_to_f32_vec(sv),
        ColumnarValue::Array(arr) => {
            // FixedSizeList — DuckDB FLOAT[N] columns
            if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
                if fsl.is_empty() {
                    return Err(DataFusionError::Execution("query vec is empty".into()));
                }
                return inner_to_f32(&*fsl.value(0), "query_vec");
            }
            // List / LargeList — Postgres real[] or float8[]
            if let Some(la) = arr.as_any().downcast_ref::<ListArray>() {
                if la.is_empty() {
                    return Err(DataFusionError::Execution("query vec is empty".into()));
                }
                return inner_to_f32(&*la.value(0), "query_vec");
            }
            if let Some(la) = arr.as_any().downcast_ref::<LargeListArray>() {
                if la.is_empty() {
                    return Err(DataFusionError::Execution("query vec is empty".into()));
                }
                return inner_to_f32(&*la.value(0), "query_vec");
            }
            // Flat Float32Array or Float64Array (uncommon but valid)
            if let Some(f32arr) = arr.as_any().downcast_ref::<Float32Array>() {
                return Ok(f32arr.values().to_vec());
            }
            if let Some(f64arr) = arr.as_any().downcast_ref::<Float64Array>() {
                return Ok(f64arr.values().iter().map(|&v| v as f32).collect());
            }
            Err(DataFusionError::Execution(format!(
                "Cannot interpret query arg as f32 vec: {:?}",
                arr.data_type()
            )))
        }
    }
}

fn scalar_to_f32_vec(sv: &ScalarValue) -> Result<Vec<f32>> {
    match sv {
        ScalarValue::FixedSizeList(arr) => inner_to_f32(&*arr.value(0), "scalar_query_vec"),
        ScalarValue::List(arr) => inner_to_f32(&*arr.value(0), "scalar_query_vec"),
        ScalarValue::LargeList(arr) => inner_to_f32(&*arr.value(0), "scalar_query_vec"),
        other => Err(DataFusionError::Execution(format!(
            "Unsupported scalar type for query vec: {:?}",
            other
        ))),
    }
}

pub fn l2_distance_udf() -> DistanceUDF {
    DistanceUDF::new("l2_distance", l2_kernel)
}
pub fn cosine_distance_udf() -> DistanceUDF {
    DistanceUDF::new("cosine_distance", cosine_kernel)
}
pub fn negative_dot_product_udf() -> DistanceUDF {
    DistanceUDF::new("negative_dot_product", negative_dot_kernel)
}
