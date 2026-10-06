//! NumPy dtypes: what a dtype argument names, and converting values into it.
//!
//! Every place that takes a dtype — `x.astype(t)`, `np.t(x)`,
//! `np.array(x, dtype=t)`, the `dtype=` argument of the array constructors —
//! parses it with [`Dtype::from_value`] and converts with [`convert`], so the
//! conversions follow NumPy (`astype` with its default unsafe casting) in one
//! place:
//!
//! * `bool`: `x != 0`; a complex value is true when either part is non-zero.
//!   Booleans are `Value::Boolean` cells, as comparison results are.
//! * `int64` (and `int`): an integer is unchanged; a float is truncated
//!   toward zero (NumPy's C cast, Python's `int()`), and a float whose
//!   truncation leaves the signed 64-bit window has no proof (the `IntCast`
//!   lowerings check the window).
//! * A fixed-width integer type of `w < 64` bits (`uint8`, `int16`, ...): the
//!   int64 value (a float truncated first) modulo `2^w`, read as two's
//!   complement for the signed types — NumPy's wrap-around. For a float this
//!   equals NumPy on x86-64 whenever the truncation fits in 32 bits, which
//!   holds for every value in Zinnia's float window (`|v| < 2^31`); NumPy
//!   warns "invalid value encountered in cast" outside it.
//! * `uint64`: the non-negative int64 values are unchanged; a negative value
//!   becomes `2^64 + x` in NumPy, which int64 cannot hold, so it has no proof.
//! * Floats of every width are Zinnia's Q32 fixed point; complex types are
//!   pairs of them. A complex value converted to a real type keeps its real
//!   part (NumPy does the same, with a `ComplexWarning`).
//!
//! The values of a fixed-width integer type equal NumPy's, but NumPy computes
//! their arithmetic in the type (modulo `2^w`) and converts values assigned
//! into their arrays; Zinnia does neither. The compiler therefore tracks those
//! values and rejects both uses (`builder/fixed_width.rs`): a converted value can be
//! compared, selected, sorted, indexed, returned, and converted again.

use crate::builder::IRBuilder;
use crate::ir_defs::IR;
use crate::types::{NumberType, Value, ZinniaType};

/// A fixed-width integer type other than int64: its width and signedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntType {
    pub bits: u32,
    pub signed: bool,
}

impl IntType {
    /// NumPy's name of the type (`uint8`, `int16`, ...).
    pub fn name(self) -> &'static str {
        match (self.bits, self.signed) {
            (8, true) => "int8",
            (8, false) => "uint8",
            (16, true) => "int16",
            (16, false) => "uint16",
            (32, true) => "int32",
            (32, false) => "uint32",
            (64, false) => "uint64",
            _ => unreachable!("IntType {self:?} is not a NumPy integer type"),
        }
    }

    /// The smallest and largest value of the type, as far as int64 holds
    /// them (`uint64` stops at `i64::MAX`).
    pub fn range(self) -> (i64, i64) {
        if self.bits == 64 {
            return (0, i64::MAX);
        }
        if self.signed {
            let half = 1i64 << (self.bits - 1);
            (-half, half - 1)
        } else {
            (0, (1i64 << self.bits) - 1)
        }
    }

    /// NumPy's conversion of the int64 value `v` into the type, when int64
    /// can hold the result (not for `uint64` of a negative value).
    pub fn wrap(self, v: i64) -> Option<i64> {
        if self.bits == 64 {
            return (v >= 0).then_some(v);
        }
        let modulus = 1i128 << self.bits;
        let r = (v as i128).rem_euclid(modulus);
        let half = modulus / 2;
        Some(if self.signed && r >= half {
            r - modulus
        } else {
            r
        } as i64)
    }
}

/// The fixed-width integer dtype of `bits` bits (not int64).
fn fixed(bits: u32, signed: bool) -> Dtype {
    Dtype::FixedInt(IntType { bits, signed })
}

/// A NumPy dtype as Zinnia computes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    Int64,
    /// A fixed-width integer type other than int64.
    FixedInt(IntType),
    /// Every float width (Q32 fixed point).
    Float,
    /// Every complex width (a pair of Q32 floats).
    Complex,
}

impl Dtype {
    /// The type a NumPy scalar-type attribute names (`np.uint8`,
    /// `np.float32`, `np.bool_`, ...).
    pub fn from_numpy_type(attribute: &str) -> Option<Dtype> {
        Some(match attribute {
            "bool_" | "bool" => Dtype::Bool,
            "int8" | "byte" => fixed(8, true),
            "uint8" | "ubyte" => fixed(8, false),
            "int16" | "short" => fixed(16, true),
            "uint16" | "ushort" => fixed(16, false),
            "int32" | "intc" => fixed(32, true),
            "uint32" | "uintc" => fixed(32, false),
            "int64" | "int_" | "intp" | "long" | "longlong" => Dtype::Int64,
            "uint64" | "uint" | "uintp" | "ulong" | "ulonglong" => fixed(64, false),
            "float16" | "float32" | "float64" | "half" | "single" | "double" | "longdouble" => {
                Dtype::Float
            }
            "complex64" | "complex128" | "complex256" | "csingle" | "cdouble" | "clongdouble" => {
                Dtype::Complex
            }
            _ => return None,
        })
    }

    /// The type a dtype string names: a NumPy type name, a Python type name,
    /// or a type code (`'u1'`, `'?'`, `'f8'`, ...), as `np.dtype` reads them
    /// (an optional byte-order prefix is ignored).
    pub fn from_name(name: &str) -> Option<Dtype> {
        let name = name.trim_start_matches(['<', '>', '=', '|']);
        if let Some(dtype) = Dtype::from_numpy_type(name) {
            return Some(dtype);
        }
        Some(match name {
            "int" => Dtype::Int64,
            "float" => Dtype::Float,
            "complex" => Dtype::Complex,
            "?" | "b1" => Dtype::Bool,
            "b" | "i1" => fixed(8, true),
            "B" | "u1" => fixed(8, false),
            "h" | "i2" => fixed(16, true),
            "H" | "u2" => fixed(16, false),
            "i" | "i4" => fixed(32, true),
            "I" | "u4" => fixed(32, false),
            "l" | "q" | "p" | "i8" => Dtype::Int64,
            "L" | "Q" | "P" | "u8" => fixed(64, false),
            "e" | "f" | "d" | "g" | "f2" | "f4" | "f8" | "f16" => Dtype::Float,
            "F" | "D" | "G" | "c8" | "c16" | "c32" => Dtype::Complex,
            _ => return None,
        })
    }

    /// The dtype a value passed as a dtype argument names: a type
    /// (`int`, `np.float32`, `bool`), or a dtype string. `None` for
    /// anything else (including `None` itself).
    pub fn from_value(value: &Value) -> Option<Dtype> {
        match value {
            Value::Class(ZinniaType::Boolean) => Some(Dtype::Bool),
            Value::Class(ZinniaType::Integer) => Some(Dtype::Int64),
            Value::Class(ZinniaType::Float) => Some(Dtype::Float),
            Value::Class(ZinniaType::Complex) => Some(Dtype::Complex),
            Value::String(s) => Dtype::from_name(&s.val),
            _ => None,
        }
    }

    /// [`Self::from_value`] for an argument that must be a dtype: a compile
    /// error naming `context` otherwise.
    pub fn required(value: &Value, context: &str) -> Dtype {
        Dtype::from_value(value).unwrap_or_else(|| {
            let shown = match value {
                Value::String(s) => format!("'{}'", s.val),
                other => other.zinnia_type().to_string(),
            };
            panic!("{context}: unsupported dtype {shown}")
        })
    }

    /// The value a program sees for this dtype (`np.uint8`, `x.dtype`): the
    /// type, or for the fixed-width integers, which Zinnia's type enum does
    /// not name, NumPy's dtype string (NumPy accepts it wherever a dtype is
    /// expected, and [`Self::from_value`] reads it back).
    pub fn type_value(self, b: &mut IRBuilder) -> Value {
        match self {
            Dtype::Bool => Value::Class(ZinniaType::Boolean),
            Dtype::Int64 => Value::Class(ZinniaType::Integer),
            Dtype::Float => Value::Class(ZinniaType::Float),
            Dtype::Complex => Value::Class(ZinniaType::Complex),
            Dtype::FixedInt(t) => b.ir_constant_str(t.name().to_string()),
        }
    }

    /// The storage of an array of this dtype.
    pub fn storage(self) -> NumberType {
        match self {
            Dtype::Bool | Dtype::Int64 | Dtype::FixedInt(_) => NumberType::Integer,
            Dtype::Float => NumberType::Float,
            Dtype::Complex => NumberType::Complex,
        }
    }
}

/// The operator NumPy applies for `op` between two bool arrays, or a bool
/// array and a bool (`builder/bool_dtype.rs`): `+` is a logical or (the
/// result stays bool, `True + True` is `True`), `-` raises NumPy's
/// `TypeError`; any other operator computes on the 0/1 values (`*` is
/// already the logical and).
pub fn operator_on_bools(op: &str) -> &str {
    match op {
        "add" => "or",
        "sub" => panic!(
            "TypeError: numpy boolean subtract, the `-` operator, is not supported, use the \
             bitwise_xor, the `^` operator, or the logical_xor function instead"
        ),
        other => other,
    }
}

/// NumPy's `TypeError` for `-m` of a bool array.
pub fn reject_bool_negative() -> ! {
    panic!(
        "TypeError: the numpy boolean negative, the `-` operator, is not supported, use the `~` \
         operator or the logical_not function instead"
    )
}

/// `x` converted into `to`, as NumPy's `astype` (see the module doc): a
/// scalar stays a scalar (`np.float64(2.5).astype(int)` is a NumPy scalar),
/// an array or a (nested) list becomes an array of the dtype.
pub fn convert(b: &mut IRBuilder, x: &Value, to: Dtype) -> Value {
    match x {
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. } => {
            convert_scalar(b, x, to)
        }
        Value::DynamicNDArray(data) => convert_runtime_shaped(b, data, to),
        Value::StaticArray { .. } | Value::List(_) | Value::Tuple(_) => convert_static(b, x, to),
        other => panic!("cannot convert a {} to a NumPy dtype", other.zinnia_type()),
    }
}

/// A scalar converted into `to`.
pub fn convert_scalar(b: &mut IRBuilder, x: &Value, to: Dtype) -> Value {
    match to {
        Dtype::Bool => to_bool(b, x),
        Dtype::Int64 => to_int64(b, x),
        Dtype::FixedInt(t) => {
            let v = to_int64(b, x);
            wrap_into(b, &v, t)
        }
        Dtype::Float => match x {
            Value::Complex { real, .. } => Value::Float(real.clone()),
            _ => b.ir_float_cast(x),
        },
        Dtype::Complex => match x {
            Value::Complex { .. } => x.clone(),
            _ => {
                let real = b.ir_float_cast(x);
                let imag = b.ir_constant_float(0.0);
                crate::helpers::value_ops::pack_complex_value(real, imag)
            }
        },
    }
}

/// `x != 0` (NumPy's and Python's truthiness of a number).
fn to_bool(b: &mut IRBuilder, x: &Value) -> Value {
    match x {
        // A bool cell of NumPy's bool dtype (`builder/bool_dtype.rs`).
        Value::Boolean(_) | Value::Integer(_) => b.bool_dtype_cell(x),
        Value::Float(s) => match s.static_val {
            Some(v) => b.ir_constant_bool(v != 0.0),
            None => {
                let zero = b.ir_constant_float(0.0);
                b.ir_not_equal_f(x, &zero)
            }
        },
        Value::Complex { real, imag } => {
            let real = to_bool(b, &Value::Float(real.clone()));
            let imag = to_bool(b, &Value::Float(imag.clone()));
            b.ir_logical_or(&real, &imag)
        }
        other => panic!("cannot convert a {} to bool", other.zinnia_type()),
    }
}

/// `x` as an int64 value: truncated toward zero from a float (the real part
/// of a complex), 0/1 from a boolean, as an integer cell (a bool's indexes
/// by position: `builder/bool_dtype.rs`). A fixed-width integer value
/// becomes an untracked copy: as an int64 it takes part in arithmetic as
/// NumPy's int64 values do (`builder/fixed_width.rs`).
pub fn to_int64(b: &mut IRBuilder, x: &Value) -> Value {
    match x {
        Value::Integer(_) if b.is_bool_cell(x) => b.int_dtype_cell(x),
        Value::Integer(_) => b.untracked_int64(x),
        Value::Boolean(_) => b.int_dtype_cell(x),
        Value::Float(_) => b.ir_int_cast(x),
        Value::Complex { real, .. } => b.ir_int_cast(&Value::Float(real.clone())),
        other => panic!("cannot convert a {} to an integer", other.zinnia_type()),
    }
}

/// The int64 value `v` converted into the fixed-width type `t`: modulo `2^w`
/// in two's complement (`uint64`: unchanged, no proof when negative).
///
/// The cheapest sound construction: no constraint when `v`'s proven range
/// ([`proven_range`]) already lies in the type (a fresh copy of `v`, so that only the converted
/// value is tracked); otherwise one constant-divisor remainder `ModI(v', 2^w)`
/// with `v' = v` (unsigned) or `v + 2^(w-1)` (signed, shifted back after).
/// The backends lower a remainder by a constant as a range-checked quotient
/// and remainder: the remainder in `[0, 2^w)`, the quotient in the range that
/// `v`'s proven range implies (the signed 64-bit window by default), which
/// pins both. `uint64` needs one non-negativity check (`AssertNonNegI`).
fn wrap_into(b: &mut IRBuilder, v: &Value, t: IntType) -> Value {
    let (lo, hi) = t.range();
    let converted = if let Some(c) = v.int_val() {
        let wrapped = t.wrap(c).unwrap_or_else(|| {
            panic!(
                "converting {c} to {}: NumPy gives 2^64 - {}, which Zinnia's int64 values cannot hold",
                t.name(),
                c.unsigned_abs()
            )
        });
        b.ir_constant_int(wrapped)
    } else if proven_range(b, v).is_some_and(|(min, max)| lo <= min && max <= hi) {
        b.fresh_copy(v)
    } else if t.bits == 64 {
        b.ir_assert_non_neg_i(v);
        b.fresh_copy(v)
    } else {
        // `ModI` directly: its contract only requires a non-zero divisor,
        // and discharging that against every visible fact (an SMT query per
        // element) is what `ir_mod_i` would add.
        let modulus = b.ir_constant_int(1i64 << t.bits);
        if t.signed {
            let half = b.ir_constant_int(1i64 << (t.bits - 1));
            let shifted = b.ir_add_i(v, &half);
            let remainder = b.create_ir(&IR::ModI, &[shifted, modulus]);
            b.ir_sub_i(&remainder, &half)
        } else {
            b.create_ir(&IR::ModI, &[v.clone(), modulus])
        }
    };
    b.track_fixed_width(&converted, t);
    converted
}

/// The range of the integer `v` that its facts, the IR's interval analysis
/// and its min/max structure prove: `IRBuilder::ask_bounds` without the SMT
/// layer, whose per-value queries over a long dependency cone (a float
/// pipeline before the conversion) take seconds each.
fn proven_range(b: &IRBuilder, v: &Value) -> Option<(i64, i64)> {
    /// How many nested selections are followed (`np.clip` is two).
    const SELECTION_DEPTH: u32 = 4;
    let (min, max) = range_of(b, v, SELECTION_DEPTH);
    (min > i64::MIN || max < i64::MAX).then_some((min, max))
}

/// [`proven_range`] with `i64::MIN` / `i64::MAX` for an unknown end.
fn range_of(b: &IRBuilder, v: &Value, depth: u32) -> (i64, i64) {
    use crate::optim::resolver::Resolver;
    let mut intervals = crate::optim::range::RangeResolver::new();
    let interval = (
        intervals
            .resolve_min_with_stmts(v, &b.stmts)
            .unwrap_or(i64::MIN),
        intervals
            .resolve_max_with_stmts(v, &b.stmts)
            .unwrap_or(i64::MAX),
    );
    let facts = v
        .value_id()
        .and_then(|vid| crate::optim::resolver::derive_bounds_from_facts(&b.facts, vid));
    let selection = (depth > 0)
        .then(|| selection_range(b, v, depth - 1))
        .flatten();
    [facts, b.deep_bounds_walk(v), selection]
        .into_iter()
        .flatten()
        .fold(interval, |(lo, hi), (other_lo, other_hi)| {
            (lo.max(other_lo), hi.min(other_hi))
        })
}

/// The range of a selection `v = SelectI(c, t, f)`: one of the ranges of `t`
/// and `f`, and for a maximum or minimum (`c` compares `t` with `f`, as
/// `np.maximum`, `np.minimum`, `np.clip` and `max()` build them) the range of
/// the larger or smaller of the two.
fn selection_range(b: &IRBuilder, v: &Value, depth: u32) -> Option<(i64, i64)> {
    let stmt = b.stmts.get(v.stmt_id()? as usize)?;
    let (IR::SelectI, &[condition, t, f]) = (&stmt.ir, stmt.arguments.as_slice()) else {
        return None;
    };
    let operand = |id: crate::types::StmtId| {
        let value_id = b.stmts[id as usize].value_id;
        Value::Integer(crate::types::ScalarValue {
            static_val: None,
            stmt_id: Some(id),
            value_id,
        })
    };
    let (t_lo, t_hi) = range_of(b, &operand(t), depth);
    let (f_lo, f_hi) = range_of(b, &operand(f), depth);
    let comparison = &b.stmts[condition as usize];
    let larger_if_true = match comparison.ir {
        IR::GtI | IR::GteI => Some(true),
        IR::LtI | IR::LteI => Some(false),
        _ => None,
    };
    let picks_larger = match (larger_if_true, comparison.arguments.as_slice()) {
        (Some(larger), &[p, q]) if (p, q) == (t, f) => Some(larger),
        (Some(larger), &[p, q]) if (p, q) == (f, t) => Some(!larger),
        _ => None,
    };
    Some(match picks_larger {
        Some(true) => (t_lo.max(f_lo), t_hi.max(f_hi)),
        Some(false) => (t_lo.min(f_lo), t_hi.min(f_hi)),
        None => (t_lo.min(f_lo), t_hi.max(f_hi)),
    })
}

/// A statically shaped array (or a nested list) converted element-wise.
fn convert_static(b: &mut IRBuilder, x: &Value, to: Dtype) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(x);
    let cells = crate::helpers::composite::flatten_composite_with_builder(b, x);
    let converted: Vec<Value> = cells
        .iter()
        .map(|cell| convert_scalar(b, cell, to))
        .collect();
    if to == Dtype::Complex {
        let (reals, imags) = converted
            .iter()
            .map(|cell| crate::helpers::value_ops::unpack_value_to_complex_parts(b, cell))
            .unzip();
        return crate::helpers::static_array::base::build_static_array_from_flat_complex(
            b, reals, imags, shape,
        );
    }
    crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        converted,
        shape,
        to.storage(),
    )
}

/// A runtime-shaped array converted element-wise: every buffer cell (padding
/// included) is converted into a fresh buffer with the same shape metadata.
fn convert_runtime_shaped(
    b: &mut IRBuilder,
    data: &crate::types::DynamicNDArrayData,
    to: Dtype,
) -> Value {
    let mut result = crate::ops::dyn_ndarray::force_contiguous(b, data);
    let cells = crate::helpers::segment::read_all_typed(
        b,
        result.segment_id,
        result.max_length(),
        result.dtype,
    );
    let converted: Vec<Value> = cells
        .iter()
        .map(|cell| convert_scalar(b, cell, to))
        .collect();
    result.dtype = to.storage();
    result.segment_id =
        crate::helpers::segment::alloc_and_write_values(b, &converted, result.dtype);
    result.value_id = crate::types::ValueId::next();
    Value::DynamicNDArray(result)
}

/// An index (a scalar, a slice bound or an index array) as int64 values:
/// a fixed-width integer index selects by its value, and the indexing
/// lowerings compute addresses from it with int64 arithmetic, which
/// `builder/fixed_width.rs` would otherwise reject. Anything else is
/// returned unchanged (no IR when nothing is tracked).
pub fn as_index(b: &mut IRBuilder, index: &Value) -> Value {
    match b.fixed_width_content_of(index) {
        Some(_) if matches!(index, Value::Integer(_)) => b.untracked_int64(index),
        Some(_) => convert(b, index, Dtype::Int64),
        None => index.clone(),
    }
}

/// `x.dtype`, as the value a program sees (see [`element_dtype`]).
pub fn dtype_of(b: &mut IRBuilder, x: &Value) -> Value {
    element_dtype(b, x).type_value(b)
}

/// The dtype of the elements of `x` (an array, a list or a scalar): a
/// fixed-width integer type when the compiler tracks them as one
/// (`builder/fixed_width.rs`), bool for a bool array or bool cells
/// (`builder/bool_dtype.rs`), else from the storage or the elements.
pub fn element_dtype(b: &mut IRBuilder, x: &Value) -> Dtype {
    if let Some(t) = b.fixed_width_content(x) {
        return Dtype::FixedInt(t);
    }
    let is_bool = match x {
        Value::Integer(_) | Value::Boolean(_) => b.is_bool_cell(x),
        _ => crate::helpers::array_ops::is_bool_array(b, x),
    };
    if is_bool {
        return Dtype::Bool;
    }
    let storage = match x {
        Value::StaticArray { dtype, .. } => *dtype,
        Value::DynamicNDArray(data) => data.dtype,
        _ => {
            let cells = crate::helpers::composite::flatten_composite(x);
            if cells.iter().any(|v| matches!(v, Value::Complex { .. })) {
                NumberType::Complex
            } else if cells.iter().any(|v| matches!(v, Value::Float(_))) {
                NumberType::Float
            } else {
                NumberType::Integer
            }
        }
    };
    match storage {
        NumberType::Integer => Dtype::Int64,
        NumberType::Float => Dtype::Float,
        NumberType::Complex => Dtype::Complex,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_follows_numpy() {
        let u8t = IntType {
            bits: 8,
            signed: false,
        };
        let i8t = IntType {
            bits: 8,
            signed: true,
        };
        let i16t = IntType {
            bits: 16,
            signed: true,
        };
        let u64t = IntType {
            bits: 64,
            signed: false,
        };
        // np.array([0, 2, -1, 300, 200, -129]).astype(np.uint8 / np.int8)
        let xs = [0i64, 2, -1, 300, 200, -129];
        let as_u8: Vec<_> = xs.iter().map(|&x| u8t.wrap(x).unwrap()).collect();
        let as_i8: Vec<_> = xs.iter().map(|&x| i8t.wrap(x).unwrap()).collect();
        assert_eq!(as_u8, [0, 2, 255, 44, 200, 127]);
        assert_eq!(as_i8, [0, 2, -1, 44, -56, 127]);
        assert_eq!(i16t.wrap(70000), Some(4464));
        assert_eq!(u64t.wrap(5), Some(5));
        assert_eq!(u64t.wrap(-1), None);
    }

    #[test]
    fn dtype_names_follow_numpy() {
        assert_eq!(
            Dtype::from_name("u1"),
            Some(Dtype::FixedInt(IntType {
                bits: 8,
                signed: false
            }))
        );
        assert_eq!(
            Dtype::from_name("<i4"),
            Some(Dtype::FixedInt(IntType {
                bits: 32,
                signed: true
            }))
        );
        assert_eq!(Dtype::from_name("int"), Some(Dtype::Int64));
        assert_eq!(Dtype::from_name("?"), Some(Dtype::Bool));
        assert_eq!(Dtype::from_name("float32"), Some(Dtype::Float));
        assert_eq!(
            Dtype::from_numpy_type("e"),
            None,
            "np.e is a constant, not a type"
        );
        assert_eq!(
            Dtype::from_numpy_type("uint"),
            Some(Dtype::FixedInt(IntType {
                bits: 64,
                signed: false
            }))
        );
    }
}
