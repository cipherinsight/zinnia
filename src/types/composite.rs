use super::envelope::Envelope;
use super::scalar::ScalarValue;
use super::value::Value;
use super::zinnia_type::{NumberType, ZinniaType};
use super::ValueId;

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

/// Storage for DynamicNDArray values.
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
    /// reads go through `storage_read_at`, writes through `storage_write_at`.
    pub segment_id: u32,
    pub meta: DynArrayMeta,
    /// Compilation-layer identity of the *whole tensor* (as opposed to
    /// any of its sub-ptrs). Minted at materialisation; preserved by
    /// `Clone` and by view-style ops that share segment storage.
    pub value_id: ValueId,
}

impl DynamicNDArrayData {
    /// Backing-buffer slot count, derived from the envelope's total bound.
    pub fn max_length(&self) -> usize {
        self.envelope.total_bound
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
///
/// Carries a `value_id` minted at construction so static composites can
/// participate in fact-anchoring (compiler.value-list-tuple-value-id).
/// No `PartialEq` impl: nothing in the codebase compares `CompositeData`
/// values for structural equality, and `Value` (the element type) is
/// not itself `PartialEq`. If equality becomes needed later, follow the
/// `ScalarValue` convention and exclude `value_id`.
#[derive(Debug, Clone)]
pub struct CompositeData {
    pub elements_type: Vec<ZinniaType>,
    pub values: Vec<Value>,
    /// Compilation-layer identity of this composite. Minted fresh at
    /// every construction site (via `ValueId::next()`); preserved across
    /// `Clone`. The preferred construction path is
    /// [`CompositeData::new`].
    pub value_id: ValueId,
}

impl CompositeData {
    /// Preferred construction path. Mints a fresh `value_id`.
    pub fn new(elements_type: Vec<ZinniaType>, values: Vec<Value>) -> Self {
        Self {
            elements_type,
            values,
            value_id: ValueId::next(),
        }
    }
}
