use super::zinnia_type::{NumberType, ZinniaType};
use super::scalar::{ScalarValue, StringValue};
use super::composite::{CompositeData, NDArrayData, DynamicNDArrayData};
use super::{StmtId, ValueId};

/// The unified value enum representing all Zinnia values during compilation.
/// Merges the Python Value hierarchy + DTDescriptor into a single tagged union.
#[derive(Debug, Clone)]
pub enum Value {
    Integer(ScalarValue<i64>),
    Float(ScalarValue<f64>),
    Boolean(ScalarValue<bool>),
    String(StringValue),
    None,
    Class(ZinniaType),
    /// Complex scalar — stored as two Float ScalarValues at the IR layer
    /// but tagged as `ZinniaType::Complex` so dispatch (arithmetic,
    /// reductions, …) can route correctly. See compiler.complex-* cards.
    Complex {
        real: ScalarValue<f64>,
        imag: ScalarValue<f64>,
    },
    NDArray(NDArrayData),
    DynamicNDArray(DynamicNDArrayData),
    /// Static-shape numeric array backed by a flat ZKRAM segment.
    ///
    /// P1 of `compiler.epic-segment-native-static-arrays`: the segment-native
    /// representation that replaces nested `Value::List` for purely-numeric
    /// arrays. Static-index access compiles to a direct wire reference, and
    /// dynamic-index access is a single zkRAM op (lands in P2/P3).
    ///
    /// Layout: row-major, the same conventions as `DynamicNDArrayData`.
    /// `offset` lets a future view variant share an underlying segment.
    ///
    /// **P5a — Complex dtype**: Complex arrays carry a *second* segment for
    /// the imaginary parts (`imag_segment_id`). Both segments share `shape`,
    /// `strides`, and `offset` and are addressed in lockstep. A single
    /// `segment_id` cannot represent two-cell-per-element layouts cheaply;
    /// using two parallel segments keeps the storage and addressing logic
    /// uniform across dtypes (one cell == one segment slot, Complex just
    /// needs two parallel segments). Non-Complex dtypes carry `None`.
    StaticArray {
        dtype: NumberType,
        shape: Vec<usize>,
        segment_id: u32,
        strides: Vec<usize>,
        offset: usize,
        /// `Some(seg)` only when `dtype == NumberType::Complex`. Holds the
        /// imaginary-part segment for dual-segment Complex arrays.
        imag_segment_id: Option<u32>,
        /// Compilation-layer identity. Fresh at every construction site
        /// (via `ValueId::next()`); preserved by Clone. View-style ops
        /// that share segment storage mint a fresh value_id by default,
        /// matching the DynamicNDArrayData and CompositeData convention.
        value_id: ValueId,
    },
    List(CompositeData),
    Tuple(CompositeData),
    PoseidonHashed {
        dtype: Box<ZinniaType>,
        inner: Box<Value>,
    },
}

impl Value {
    /// Returns the `ZinniaType` of this value.
    pub fn zinnia_type(&self) -> ZinniaType {
        match self {
            Value::Integer(_) => ZinniaType::Integer,
            Value::Float(_) => ZinniaType::Float,
            Value::Boolean(_) => ZinniaType::Boolean,
            Value::String(_) => ZinniaType::String,
            Value::None => ZinniaType::None,
            Value::Class(inner_type) => inner_type.clone(),
            Value::Complex { .. } => ZinniaType::Complex,
            Value::NDArray(data) => ZinniaType::NDArray {
                shape: data.shape.clone(),
                dtype: data.dtype,
            },
            Value::DynamicNDArray(data) => ZinniaType::DynamicNDArray {
                dtype: data.dtype,
                max_length: data.max_length(),
                max_rank: data.max_rank(),
            },
            Value::StaticArray { dtype, shape, .. } => ZinniaType::NDArray {
                shape: shape.clone(),
                dtype: *dtype,
            },
            Value::List(data) => ZinniaType::List {
                elements: data.elements_type.clone(),
            },
            Value::Tuple(data) => ZinniaType::Tuple {
                elements: data.elements_type.clone(),
            },
            Value::PoseidonHashed { dtype, .. } => ZinniaType::PoseidonHashed {
                dtype: dtype.clone(),
            },
        }
    }

    /// Returns the IR statement ID for atomic values, if available.
    /// (Phase 2 — renamed from `ptr` to disambiguate IR-statement identity
    /// from the C/C++ "pointer" connotation.)
    pub fn stmt_id(&self) -> Option<StmtId> {
        match self {
            Value::Integer(s) => s.stmt_id,
            Value::Float(s) => s.stmt_id,
            Value::Boolean(s) => s.stmt_id,
            Value::String(s) => Some(s.stmt_id),
            Value::None => Option::None,
            Value::Class(_) => Option::None,
            _ => Option::None,
        }
    }

    /// Returns the compilation-layer identity of this Value, if defined
    /// (compiler.value-id-and-fact-leaves). Scalars and dyn ndarrays
    /// carry one; pure-type or pure-name Values (None / Class / String /
    /// composites without an explicit identity) return `None`.
    pub fn value_id(&self) -> Option<ValueId> {
        match self {
            Value::Integer(s) => Some(s.value_id),
            Value::Float(s) => Some(s.value_id),
            Value::Boolean(s) => Some(s.value_id),
            Value::DynamicNDArray(d) => Some(d.value_id),
            // Composite scalars: complex carries two scalar Values; we
            // expose only the real part's identity (analogous to what
            // `stmt_id()` does for similar composites).
            Value::Complex { real, .. } => Some(real.value_id),
            Value::List(d) | Value::Tuple(d) => Some(d.value_id),
            Value::StaticArray { value_id, .. } => Some(*value_id),
            _ => None,
        }
    }

    /// Returns true if this value is a numeric type.
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Integer(_) | Value::Float(_) | Value::Boolean(_))
    }

    /// Returns true if this value is an integer-like type.
    pub fn is_integer(&self) -> bool {
        matches!(self, Value::Integer(_) | Value::Boolean(_))
    }

    /// Returns true if this value is an NDArray or DynamicNDArray.
    pub fn is_ndarray(&self) -> bool {
        matches!(
            self,
            Value::NDArray(_) | Value::DynamicNDArray(_) | Value::StaticArray { .. }
        )
    }

    /// Returns the compile-time integer value, if known.
    pub fn int_val(&self) -> Option<i64> {
        match self {
            Value::Integer(s) => s.static_val,
            Value::Boolean(s) => s.static_val.map(|b| if b { 1 } else { 0 }),
            _ => Option::None,
        }
    }

    /// Returns the compile-time float value, if known.
    pub fn float_val(&self) -> Option<f64> {
        match self {
            Value::Float(s) => s.static_val,
            _ => Option::None,
        }
    }

    /// Returns the compile-time bool value, if known.
    pub fn bool_val(&self) -> Option<bool> {
        match self {
            Value::Boolean(s) => s.static_val,
            _ => Option::None,
        }
    }

    /// Returns the compile-time string value, if known.
    pub fn string_val(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(&s.val),
            _ => Option::None,
        }
    }
}
