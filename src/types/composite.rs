use super::envelope::Envelope;
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
///
/// The compile-time shape envelope (per-axis Static/Dynamic bounds with
/// symbolic dim variables for unification) lives in [`Envelope`]. The
/// previous `(max_length, max_rank)` fields have been replaced — callers
/// can recover them via [`DynamicNDArrayData::max_length`] and
/// [`DynamicNDArrayData::max_rank`].
#[derive(Debug, Clone)]
pub struct DynamicNDArrayData {
    /// Compile-time shape envelope (per-axis bounds + dim variables).
    pub envelope: Envelope,
    pub dtype: NumberType,
    /// ZKRAM segment ID. Every dynamic ndarray is backed by a segment;
    /// reads go through `ir_read_memory`, writes through `ir_write_memory`.
    pub segment_id: u32,
    pub meta: DynArrayMeta,
}

impl DynamicNDArrayData {
    /// Worst-case total element count, derived from the envelope.
    pub fn max_length(&self) -> usize {
        self.envelope.max_total()
    }

    /// Compile-time fixed rank, derived from the envelope.
    pub fn max_rank(&self) -> usize {
        self.envelope.rank()
    }
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
