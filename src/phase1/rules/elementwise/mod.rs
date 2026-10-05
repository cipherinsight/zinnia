//! Rule group `elementwise` (`revision/phase1/rules/elementwise/README.md`).
//!
//! The nodes that apply a scalar primitive position-wise. The group moves no
//! position and introduces no count; it decides how the operands' positions
//! are lined up — aligned slots (`MAP_ALIGNED`), broadcast reads
//! (`BROADCAST_MAP`), or per-entry fetches (`ELEMENTWISE_BATCH`) — and emits
//! one gadget per slot or entry.
//!
//! This module holds what the three files share: the source pattern, the
//! primitive table and the gadget costs of `map_aligned.md`, and the λ an
//! output carries, which `../README.md` §4.1 reads off the operands.

pub(crate) mod broadcast_map;
pub(crate) mod elementwise_batch;
pub(crate) mod map_aligned;

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::{bits_for, Builder};
use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::interface::Layout;
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::{Condition, Question};
use crate::phase1::rules::Candidate;

/// What the source pattern of the three rules read (`map_aligned.md`).
#[derive(Debug, Clone)]
pub struct Match {
    /// The primitive, in the front end's spelling.
    pub primitive: String,
    /// The operands of rank ≥ 1, in operand order.
    pub arrays: Vec<ValueKey>,
    /// The operands of rank 0, in operand order.
    pub scalars: Vec<ValueKey>,
    pub dtype: Dtype,
    /// The dtype the gadget computes in, which is the operands' rather than
    /// the output's: a comparison on Float elements gives a Bool, and its
    /// row is `fop⟨lt_f⟩`, not an integer `cmp_w`.
    pub operand_dtype: Dtype,
    pub dims: Vec<TermId>,
    /// Whether the call wrote `atol=0`, which is what makes `isclose` on
    /// integer operands an equality (`map_aligned.md`, the `isclose` row).
    pub exact: bool,
    /// The literal exponent of `pow` on the integer family, when CONST
    /// folded the second operand to one (`map_aligned.md`, the `power` row).
    /// The rule fills it in; the source pattern leaves it `None`.
    pub exponent: Option<i64>,
    /// The dtype the value operands promote to, with the result's dtype left
    /// out: what a cast reads *from*. `NA.astype(float)` has `operand_dtype`
    /// Float, because that is what the gadget computes in, and
    /// `source_dtype` I64, which is what makes the row a `float_cast`
    /// (`map_aligned.md`, the cast rows).
    pub source_dtype: Option<Dtype>,
}

/// The phase 0 transfer rules the group covers.
const COVERED: [Phase0Rule; 5] = [
    Phase0Rule::EMap,
    Phase0Rule::EZip,
    Phase0Rule::ECmp,
    Phase0Rule::EWhere3,
    Phase0Rule::EOuter,
];

/// `source_pattern` of `map_aligned.md`: the AST node names the primitive
/// and the producer names the operands. The result is an array of rank ≥ 1
/// and at least one operand is an array operand (S1).
pub fn matches(builder: &Builder<'_>, key: &ValueKey) -> Option<Match> {
    let array = builder.entry(key)?.desc.as_array()?;
    if array.rank() == 0 {
        return None;
    }
    let matched = classify(builder, key, array.dtype, array.dims.clone())?;
    // S1: at least one array operand.
    (!matched.arrays.is_empty()).then_some(matched)
}

/// The rank-0 form of [`matches`] (`elementwise_batch.md`, the `scalar`
/// variant): an elementwise operation whose result has rank 0 and every
/// operand of which is therefore a scalar operand.
///
/// Such a result is a scalar, so its row is the `Wire` row of
/// `rules/README.md` §3.2 rather than a bundle family:
/// `np.sin(np.deg2rad(degree))` on a scalar `degree`, or
/// `min(2 * (bound - 1) - idx, max(idx, -idx))` on Float scalars.
pub fn matches_scalar(builder: &Builder<'_>, key: &ValueKey) -> Option<Match> {
    let dtype = match &builder.entry(key)?.desc {
        crate::phase0::desc::Desc::Scalar(scalar) => scalar.dtype,
        crate::phase0::desc::Desc::Array(array) if array.rank() == 0 => array.dtype,
        _ => return None,
    };
    let matched = classify(builder, key, dtype, Vec::new())?;
    // A rank-0 result has no array operand: every operand is read as a wire.
    matched.arrays.is_empty().then_some(matched)
}

/// The operands and the primitive of one elementwise value, shared by
/// [`matches`] and [`matches_scalar`]. `dtype` and `dims` are the result's,
/// which the two forms read off different descriptors.
fn classify(
    builder: &Builder<'_>,
    key: &ValueKey,
    dtype: Dtype,
    dims: Vec<TermId>,
) -> Option<Match> {
    let entry = builder.entry(key)?;
    let Producer::Applied { rules, operands, .. } = &entry.producer else {
        return None;
    };
    if !rules.iter().any(|rule| COVERED.contains(rule)) {
        return None;
    }
    // A value that is also a view or a reduction is another group's.
    if rules.iter().any(|rule| !COVERED.contains(rule)) {
        return None;
    }
    if !super::sources::has_encoding(dtype) {
        return None;
    }
    let node = builder.node(key)?;
    let primitive = node
        .operator()
        .map(str::to_string)
        .or_else(|| node.member().map(str::to_string))?;
    let mut arrays = Vec::new();
    let mut scalars = Vec::new();
    // The dtype the operands promote to, which is what the gadget computes
    // in. `where`'s first operand is the Boolean mask, not a value the
    // gadget reads, so the output's dtype stands in for it.
    let mut operand_dtype = dtype;
    let mut source_dtype: Option<Dtype> = None;
    for operand in operands {
        let promote = |current: Dtype, other: Dtype| {
            if primitive == "where" {
                current
            } else {
                Dtype::promote(current, other).unwrap_or(current)
            }
        };
        let widen = |current: Option<Dtype>, other: Dtype| match current {
            None => Some(other),
            Some(held) => Some(Dtype::promote(held, other).unwrap_or(held)),
        };
        match builder.entry(operand).map(|entry| &entry.desc) {
            // The dtype argument of a cast — `a.astype(float)`,
            // `np.int64(x)` — is a `Class` descriptor. It names the target
            // dtype, which the result's own descriptor already carries, and
            // no value the gadget reads. Phase 0 puts it at a position that
            // depends on how the call was written — before the receiver for
            // `name.astype(t)`, after it for `expr.astype(t)` — so it is
            // skipped by kind and not by position
            // (`elementwise/README.md`, `source_pattern`).
            Some(crate::phase0::desc::Desc::Class(_)) => continue,
            Some(crate::phase0::desc::Desc::Array(operand_array)) => {
                operand_dtype = promote(operand_dtype, operand_array.dtype);
                source_dtype = widen(source_dtype, operand_array.dtype);
                if operand_array.rank() == 0 {
                    scalars.push(operand.clone());
                } else {
                    arrays.push(operand.clone());
                }
            }
            Some(crate::phase0::desc::Desc::Scalar(operand_scalar)) => {
                operand_dtype = promote(operand_dtype, operand_scalar.dtype);
                source_dtype = widen(source_dtype, operand_scalar.dtype);
                scalars.push(operand.clone());
            }
            _ => return None,
        }
    }
    if !super::sources::has_encoding(operand_dtype) {
        return None;
    }
    let exact = node
        .parameter(3, "atol")
        .and_then(|value| value.int())
        .map(|value| value == 0)
        .unwrap_or(false);
    Some(Match {
        primitive,
        arrays,
        scalars,
        dtype,
        operand_dtype,
        dims,
        exact,
        exponent: None,
        source_dtype,
    })
}

/// The gadget of one row of `map_aligned.md`'s primitive table, in the
/// primitives of `../README.md` §7, for one position.
///
/// `None` for an operation the table states no row for, which drops the node
/// and is reported with the operator's name.
pub fn gadget(builder: &Builder<'_>, matched: &Match) -> Option<Vec<(Primitive, i128)>> {
    let (primitive, dtype) = (matched.primitive.as_str(), matched.operand_dtype);
    if super::sources::is_float(dtype) {
        return float_gadget(primitive, matched.source_dtype);
    }
    let width = builder.omega.round(builder.dtype_width(dtype));
    let wraps = dtype != Dtype::Bool;
    // The carry of a sum or a difference is 2 bits, of a product w.
    let wrap = |carry: u32| Primitive::Wrap { width, carry_width: builder.omega.round(carry) };
    let comparison = Primitive::Cmp { width };
    Some(match primitive {
        "add" | "sub" | "subtract" | "positive" | "uadd" if dtype == Dtype::Bool => {
            vec![(Primitive::Or, 1)]
        }
        "add" | "sub" | "subtract" | "usub" | "negative" => {
            if wraps {
                vec![(wrap(2), 1)]
            } else {
                Vec::new()
            }
        }
        "mul" | "multiply" | "square" if dtype == Dtype::Bool => vec![(Primitive::And, 1)],
        "mul" | "multiply" | "square" => vec![(Primitive::Mul, 1), (wrap(width), 1)],
        "floor_div" | "floor_divide" | "mod" | "fmod" | "remainder" => vec![
            (Primitive::Eq, 1),
            (comparison, 1),
            (Primitive::Mul, 7),
            (Primitive::Range { width }, 2),
            (Primitive::Range { width: builder.omega.round(width + 1) }, 1),
            (wrap(2), 1),
        ],
        "lt" | "less" | "gt" | "greater" | "lte" | "less_equal" | "gte" | "greater_equal" => {
            vec![(comparison, 1)]
        }
        "eq" | "equal" | "ne" | "not_equal" if dtype == Dtype::Bool => vec![(Primitive::Xor, 1)],
        "eq" | "equal" | "ne" | "not_equal" => vec![(Primitive::Eq, 1)],
        "and" | "logical_and" | "bit_and" if dtype == Dtype::Bool => vec![(Primitive::And, 1)],
        "or" | "logical_or" | "bit_or" if dtype == Dtype::Bool => vec![(Primitive::Or, 1)],
        "bit_xor" | "logical_xor" if dtype == Dtype::Bool => vec![(Primitive::Xor, 1)],
        "not" | "invert" | "logical_not" if dtype == Dtype::Bool => vec![(Primitive::Not, 1)],
        "invert" => Vec::new(),
        "bit_and" | "bit_or" | "bit_xor" => vec![
            (Primitive::Range { width }, 1),
            (Primitive::Mul, width as i128),
        ],
        "min" | "minimum" | "max" | "maximum" | "fmin" | "fmax" => {
            vec![(comparison, 1), (Primitive::Select, 1)]
        }
        "clip" => vec![(comparison, 2), (Primitive::Select, 2)],
        "abs" | "absolute" | "fabs" => {
            vec![(comparison, 1), (Primitive::Select, 1), (wrap(2), 1)]
        }
        "sign" => vec![(comparison, 2)],
        "where" => vec![(Primitive::Select, 1)],
        // `imag` of a real is the constant 0, `real` and `conjugate` the
        // identity: no gate either way (`map_aligned.md`, the last rows).
        "floor" | "ceil" | "trunc" | "round" | "real" | "imag" | "conjugate" => Vec::new(),
        // The casts of the table: an integer widening is free, `int → Bool`
        // is `not(eq(a, 0))`.
        "astype" | "int" | "int64" | "long" => Vec::new(),
        "bool" | "bool_" => vec![(Primitive::Eq, 1), (Primitive::Not, 1)],
        // `np.isclose(a, b, atol=0)` on integer operands is `a = b`: the
        // difference is an integer, so `|a − b| ≤ rtol·|b|` holds exactly
        // when it is zero, which is the front end's own specialisation
        // (`ir_gen/named_attr/numpy_funcs.rs`, `np_isclose`). A non-zero
        // tolerance on integers has no row, and the node is dropped.
        "isclose" if matched.exact => vec![(Primitive::Eq, 1)],
        // `a ** e` for a literal `e ≥ 1` is square-and-multiply: `n_e`
        // products, each a `mul` wrapped at the dtype boundary. `e = 0` is
        // the constant 1 and costs nothing; a non-literal or negative
        // exponent has no row (`map_aligned.md`, the `power` row).
        "pow" | "power" => match matched.exponent {
            Some(0) => Vec::new(),
            Some(exponent) if exponent >= 1 => {
                let products = square_and_multiply(exponent as u64);
                vec![(Primitive::Mul, products), (wrap(width), products)]
            }
            _ => return None,
        },
        "shl" | "left_shift" => vec![(wrap(width + 1), 1)],
        "shr" | "right_shift" => vec![(
            Primitive::DivModConst {
                quotient_width: width,
                divisor_width: builder.omega.round(bits_for(2)),
                power_of_two: true,
            },
            1,
        )],
        _ => return None,
    })
}

/// The literal the `power` row needs and the AST node does not carry.
///
/// `map_aligned.md` lists "the exponent of `pow`" among the constant
/// parameters, but the front end writes `a ** e` as a binary operator with
/// the exponent as its second operand, so the literal is the CONST answer on
/// that operand's term. The question is asked once per node and recorded in
/// A like every other. A Float `pow` needs no literal: its row is one
/// `fop⟨pow_f⟩` whatever the exponent.
pub fn resolve_exponent(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    matched: &mut Match,
    epoch: u32,
    under: Condition,
) {
    if !matches!(matched.primitive.as_str(), "pow" | "power")
        || super::sources::is_float(matched.operand_dtype)
    {
        return;
    }
    let Some(operand) = matched.scalars.last().cloned() else {
        return;
    };
    let Some(term) = builder.scalar(&operand).and_then(|scalar| scalar.term.as_int()) else {
        return;
    };
    if let Verdict::Constant(value) =
        builder.ask_for(candidate, Question::Const(term), epoch, under)
    {
        matched.exponent = i64::try_from(value).ok();
    }
}

/// `n_e = ⌊log₂ e⌋ + popcount(e) − 1`, the products square-and-multiply
/// takes for a literal exponent `e ≥ 1` (`map_aligned.md`, the `power` row).
fn square_and_multiply(exponent: u64) -> i128 {
    let squarings = 63 - exponent.leading_zeros();
    (squarings + exponent.count_ones() - 1) as i128
}

/// The Float rows of `map_aligned.md`'s primitive table, one position at a
/// time: the `fop⟨name⟩`s of `../README.md` §7 the backend's own float
/// vocabulary names (`src/ir_defs.rs`).
///
/// A Float wire has no width, so no row here carries a `range_w`, a `wrap_w`
/// or a decomposition; a comparison gives a Bool through its `fop`, and
/// `where` is the one row that is not a float instruction at all — a
/// `select_f` on the mask's wire.
///
/// `None` for an operation the backend has no float instruction for
/// (`arctan`, the bitwise operations, a shift), which drops the node and is
/// reported with the operator's name.
pub fn float_gadget(
    primitive: &str,
    source_dtype: Option<Dtype>,
) -> Option<Vec<(Primitive, i128)>> {
    use FloatOp::*;
    let one = |op: FloatOp| vec![(Primitive::Float(op), 1)];
    let many = |ops: &[FloatOp]| -> Vec<(Primitive, i128)> {
        ops.iter().map(|&op| (Primitive::Float(op), 1)).collect()
    };
    Some(match primitive {
        "add" | "uadd" | "positive" => one(Add),
        "sub" | "subtract" => one(Sub),
        // `−x` is `0 − x`, which is what the front end emits.
        "usub" | "negative" => one(Sub),
        "mul" | "multiply" | "square" => one(Mul),
        "div" | "divide" | "true_divide" => one(Div),
        "reciprocal" => one(Div),
        "floor_div" | "floor_divide" => one(FloorDiv),
        "mod" | "fmod" | "remainder" => one(Mod),
        "pow" | "power" => one(Pow),
        "abs" | "absolute" | "fabs" => one(Abs),
        "sign" => one(Sign),
        "eq" | "equal" => one(Eq),
        "ne" | "not_equal" => one(Ne),
        "lt" | "less" => one(Lt),
        "lte" | "less_equal" => one(Lte),
        "gt" | "greater" => one(Gt),
        "gte" | "greater_equal" => one(Gte),
        // `min(a, b)` is `select_f(a < b, a, b)`; the backend has no float
        // minimum instruction.
        "min" | "minimum" | "fmin" | "max" | "maximum" | "fmax" => many(&[Lt, Select]),
        "clip" => many(&[Lt, Select, Gt, Select]),
        "where" => one(Select),
        "sqrt" => one(Sqrt),
        "exp" => one(Exp),
        "log" => one(Log),
        "sin" => one(Sin),
        "cos" => one(Cos),
        "tan" => one(Tan),
        "sinh" => one(SinH),
        "cosh" => one(CosH),
        "tanh" => one(TanH),
        "arcsin" | "asin" => one(ArcSin),
        "arccos" | "acos" => one(ArcCos),
        "arctan2" | "atan2" => one(ArcTan2),
        // A degree/radian conversion is a product by a literal.
        "deg2rad" | "radians" | "rad2deg" | "degrees" => one(Mul),
        // The rounding family, exactly as the front end composes it
        // (`ops/static_ndarray_ops/elementwise.rs`): `floor(x)` is
        // `floor_divide_f(x, 1)`, `ceil(x)` is `−floor(−x)`, `trunc` selects
        // between the two on the sign, `round` is `floor(±x + 0.5)`.
        "floor" => one(FloorDiv),
        "ceil" => many(&[Sub, FloorDiv, Sub]),
        "trunc" => many(&[FloorDiv, Sub, FloorDiv, Sub, Gte, Select]),
        "round" => many(&[Add, FloorDiv, Sub, Add, FloorDiv, Sub, Gte, Select]),
        // `np.isclose(a, b)` is `|a − b| ≤ atol + rtol·|b|`
        // (`ir_gen/named_attr/numpy_funcs.rs`, `np_isclose`).
        "isclose" => many(&[Sub, Abs, Abs, Mul, Add, Lte]),
        // Casts. `astype`/`float` on a float is the same wire; an integer
        // read as a float is a `float_cast`, a float read as an integer an
        // `int_cast`.
        "float" | "float64" | "float32" | "astype" => match source_dtype {
            Some(source) if !super::sources::is_float(source) => one(FromInt),
            _ => Vec::new(),
        },
        "int" | "int64" | "long" => one(ToInt),
        // `bool(x)` on a float is `x ≠ 0`.
        "bool" | "bool_" => one(Ne),
        "real" | "conjugate" | "conj" => Vec::new(),
        // `imag` of a Float is the constant 0.0, which costs no gate.
        "imag" => Vec::new(),
        _ => return None,
    })
}

/// The λ an elementwise output carries: the first array port's, with the
/// root and the lineage replaced by the output's own when the layout is
/// dense (`map_aligned.md`, `produced_output_bundle`).
pub fn output_layout(base: &Layout, value: &ValueKey) -> Layout {
    if base.layout.is_dense() {
        return Layout::dense(base.layout.clone(), value.clone());
    }
    base.clone()
}
