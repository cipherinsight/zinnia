use super::zinnia_type::{NumberType, ZinniaType};
use super::scalar::ScalarValue;
use super::value::Value;


/// Flat storage for NDArray values.
#[derive(Debug, Clone)]
pub struct NDArrayData {
    pub shape: Vec<usize>,
    pub dtype: NumberType,
    /// Flat storage of element values. Length = product of shape.
    pub elements: Vec<ScalarValue<i64>>,
}

impl NDArrayData {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

// ---------------------------------------------------------------------------
// DynArrayMeta — metadata for DynamicNDArray
// ---------------------------------------------------------------------------

/// Runtime metadata for dynamic-shaped arrays.
#[derive(Debug, Clone)]
pub struct DynArrayMeta {
    pub logical_shape: Vec<usize>,
    pub logical_offset: usize,
    pub logical_strides: Vec<usize>,
    pub runtime_length: ScalarValue<i64>,
    pub runtime_rank: ScalarValue<i64>,
    pub runtime_shape: Vec<ScalarValue<i64>>,
    pub runtime_strides: Vec<ScalarValue<i64>>,
    pub runtime_offset: ScalarValue<i64>,
}

// ---------------------------------------------------------------------------
// DynamicNDArrayData — storage for runtime-shaped arrays
// ---------------------------------------------------------------------------

/// Storage for DynamicNDArray values (extends NDArrayData with metadata).
#[derive(Debug, Clone)]
pub struct DynamicNDArrayData {
    pub max_length: usize,
    pub max_rank: usize,
    pub dtype: NumberType,
    /// Flat storage of element values. Length = max_length.
    pub elements: Vec<ScalarValue<i64>>,
    pub meta: DynArrayMeta,
}

// ---------------------------------------------------------------------------
// CompositeData — storage for List/Tuple values
// ---------------------------------------------------------------------------

/// Storage for List and Tuple composite values.
#[derive(Debug, Clone)]
pub struct CompositeData {
    pub elements_type: Vec<ZinniaType>,
    pub values: Vec<Value>,
}
