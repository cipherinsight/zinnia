//! Scalars: `definition.md` §5.2.
//!
//! `E-Len`, `E-Shape`, `E-Arith-Shape`, `E-Arith-Data`, `E-Cmp` and
//! `E-Elem`. `E-Reduce-Scalar`, also in §5.2, lives in
//! [`super::reductions`] next to `E-Reduce-Axis`, with which it shares
//! [`super::reductions::ReduceKind`].

use super::helpers::in_bounds;
use super::Rule;
use super::{require_int_term, RuleCtx, VarRange};
use crate::phase0::desc::{ArrayDesc, Desc, Dtype, Origin, ScalarDesc, ScalarTerm, ValueKey};
use crate::phase0::term::{BoolId, TermId};
use crate::phase0::Reject;

/// The scalar arithmetic operators. §5.2's `E-Arith-Shape` covers
/// `{+, -, *, //, %}`; `Div` (Python `/`, which yields a float) and `Pow`
/// always take the `E-Arith-Data` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    Div,
    Pow,
}

impl ArithOp {
    /// Whether the operator can fail on a zero right-hand side.
    fn divides(self) -> bool {
        matches!(self, ArithOp::FloorDiv | ArithOp::Mod)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// `E-Len` (§5.2): `len(a)`.
///
/// For an array this is `dims_a[0]`, a shape-origin scalar. For a composite
/// it is the static length (`E-Property`, §5.11). The baseline returns `0`
/// for anything else (`coverage.md` §9, a defect); phase 0 rejects instead.
pub fn e_len(ctx: &mut RuleCtx, a: &Desc) -> Result<Desc, Reject> {
    let t = match a {
        Desc::Array(arr) => *arr.dims.first().ok_or_else(|| Reject::Premise {
            rule: Rule::ELen,
            detail: "of a rank-0 array".to_string(),
        })?,
        Desc::Composite(c) => ctx.terms.constant(c.elems.len() as i128),
        other => {
            return Err(Reject::Premise {
                rule: Rule::ELen,
                detail: format!("of a {}", other.kind_str()),
            })
        }
    };
    let tt = ctx.tt();
    Ok(ctx.scalar(
        Rule::ELen,
        Dtype::Idx,
        ScalarTerm::Int(t),
        Origin::Shape,
        tt,
    ))
}

/// `E-Shape` (§5.2): `a.shape[k]` with `k` a constant.
pub fn e_shape(ctx: &mut RuleCtx, a: &ArrayDesc, k: i64) -> Result<Desc, Reject> {
    let axis = super::helpers::normalise_axis(Rule::EShape, k, a.rank())?;
    let t = a.dims[axis];
    let tt = ctx.tt();
    Ok(ctx.scalar(
        Rule::EShape,
        Dtype::Idx,
        ScalarTerm::Int(t),
        Origin::Shape,
        tt,
    ))
}

/// `E-Arith-Shape` / `E-Arith-Data` (§5.2): `s1 op s2`.
///
/// The shape rule applies when both operands have origin `shape` and the
/// premise holds (`*` needs a constant side; `//` and `%` need a non-zero
/// constant right side). Otherwise the data rule applies: a fresh variable
/// with the dtype range, and the arithmetic relation dropped — a deliberate
/// loss, because keeping it would make $\Phi$ record arbitrary scalar
/// computation (§3.2).
/// `nonzero` is the Boolean the caller prepared for "the divisor is not
/// zero" where the divisor carries no term to compare with 0 (§5.2's
/// `defcond_div`); it is `None` for every operator that cannot divide.
pub fn e_arith(
    ctx: &mut RuleCtx,
    op: ArithOp,
    lhs: &ScalarDesc,
    rhs: &ScalarDesc,
    value: &ValueKey,
    base: &str,
    nonzero: Option<BoolId>,
) -> Result<Desc, Reject> {
    let dtype = Dtype::promote(lhs.dtype, rhs.dtype).ok_or_else(|| Reject::Premise {
        rule: Rule::EArithData,
        detail: format!("on {} and {}", lhs.dtype.as_str(), rhs.dtype.as_str()),
    })?;
    if lhs.origin == Origin::Shape && rhs.origin == Origin::Shape {
        if let Some(desc) = e_arith_shape(ctx, op, lhs, rhs, dtype) {
            return Ok(desc);
        }
    }
    Ok(e_arith_data(ctx, op, rhs, dtype, value, base, nonzero))
}

/// `E-Arith-Shape` (§5.2). `None` when the premise fails, which sends the
/// caller to `E-Arith-Data`.
fn e_arith_shape(
    ctx: &mut RuleCtx,
    op: ArithOp,
    lhs: &ScalarDesc,
    rhs: &ScalarDesc,
    dtype: Dtype,
) -> Option<Desc> {
    let (a, b) = (lhs.term.as_int()?, rhs.term.as_int()?);
    let kb = ctx.terms.constant_value(b);
    let raw = match op {
        ArithOp::Add => ctx.terms.add(a, b),
        ArithOp::Sub => ctx.terms.sub(a, b),
        ArithOp::Mul => match (ctx.terms.constant_value(a), kb) {
            (Some(k), _) => ctx.terms.mul_const(k, b),
            (_, Some(k)) => ctx.terms.mul_const(k, a),
            _ => return None,
        },
        ArithOp::FloorDiv => ctx.terms.div_const(a, kb?)?,
        ArithOp::Mod => ctx.terms.mod_const(a, kb?)?,
        ArithOp::Div | ArithOp::Pow => return None,
    };
    let t = if dtype.wraps() {
        ctx.terms.wrap64(raw)
    } else {
        raw
    };
    // The premise made the divisor a non-zero constant, so `defcond` is true.
    let tt = ctx.tt();
    Some(ctx.scalar(
        Rule::EArithShape,
        dtype,
        ScalarTerm::Int(t),
        Origin::Shape,
        tt,
    ))
}

/// `E-Arith-Data` (§5.2): a fresh variable with the dtype range.
///
/// `defcond_div` is `t2 ≠ 0` for `//` and `%` on `Idx`; on `I64` it is true,
/// where NumPy returns 0 on division by zero. A divisor that carries no term
/// at all — a `Float` or `Complex` scalar — is compared with 0 by the data
/// Boolean the caller prepared, so that the condition still enters $\Phi$
/// and the circuit enforces it.
fn e_arith_data(
    ctx: &mut RuleCtx,
    op: ArithOp,
    rhs: &ScalarDesc,
    dtype: Dtype,
    value: &ValueKey,
    base: &str,
    nonzero: Option<BoolId>,
) -> Desc {
    let defcond = match (op.divides(), dtype, rhs.term.as_int()) {
        (true, Dtype::Idx, Some(b)) => {
            let zero = ctx.terms.constant(0);
            let is_zero = ctx.terms.eq(b, zero);
            ctx.terms.not(is_zero)
        }
        (true, _, None) => nonzero.unwrap_or_else(|| ctx.tt()),
        _ => ctx.tt(),
    };
    let range = VarRange::of_dtype(ctx.terms, dtype);
    ctx.data_scalar(Rule::EArithData, dtype, base, value, range, defcond)
}

/// `E-Cmp` (§5.2): `s1 cmp s2`.
///
/// Both operands of origin `shape` give a shape-origin Boolean whose term is
/// the comparison itself; otherwise a fresh Boolean variable $\beta$.
pub fn e_cmp(
    ctx: &mut RuleCtx,
    op: CmpOp,
    lhs: &ScalarDesc,
    rhs: &ScalarDesc,
    value: &ValueKey,
    base: &str,
) -> Desc {
    let shape_terms = if lhs.origin == Origin::Shape && rhs.origin == Origin::Shape {
        lhs.term.as_int().zip(rhs.term.as_int())
    } else {
        None
    };
    let tt = ctx.tt();
    match shape_terms {
        Some((a, b)) => {
            let t = compare(ctx, op, a, b);
            ctx.scalar(
                Rule::ECmp,
                Dtype::Bool,
                ScalarTerm::Bool(t),
                Origin::Shape,
                tt,
            )
        }
        None => {
            let range = VarRange::default();
            ctx.data_scalar(Rule::ECmp, Dtype::Bool, base, value, range, tt)
        }
    }
}

fn compare(ctx: &mut RuleCtx, op: CmpOp, a: TermId, b: TermId) -> crate::phase0::term::BoolId {
    match op {
        CmpOp::Eq => ctx.terms.eq(a, b),
        CmpOp::Ne => {
            let e = ctx.terms.eq(a, b);
            ctx.terms.not(e)
        }
        CmpOp::Lt => ctx.terms.lt(a, b),
        CmpOp::Le => ctx.terms.le(a, b),
        CmpOp::Gt => ctx.terms.lt(b, a),
        CmpOp::Ge => ctx.terms.le(b, a),
    }
}

/// `E-Elem` (§5.2): `a[i_0, …, i_{r-1}]`, all integer scalars.
///
/// The result is a data scalar. `defcond = ⋀_k (-dims_a[k] ≤ t_{i_k} <
/// dims_a[k])` enters $\Phi$ even if the element is never used afterwards:
/// an out-of-range index makes the whole run unprovable (§5.2).
pub fn e_elem(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    indices: &[&ScalarDesc],
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    if indices.len() != a.rank() {
        return Err(Reject::Premise {
            rule: Rule::EElem,
            detail: format!(
                "with {} indices on a rank-{} array",
                indices.len(),
                a.rank()
            ),
        });
    }
    let mut conditions = Vec::with_capacity(indices.len());
    for (axis, index) in indices.iter().enumerate() {
        let t = require_int_term(Rule::EElem, index)?;
        conditions.push(in_bounds(ctx.terms, t, a.dims[axis]));
    }
    let defcond = ctx.terms.all(&conditions);
    let range = VarRange::of_dtype(ctx.terms, a.dtype);
    Ok(ctx.data_scalar(Rule::EElem, a.dtype, base, value, range, defcond))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::tests::Fixture;

    /// §5.2: `len(a)` is `dims_a[0]`, `a.shape[k]` is `dims_a[k]`, both of
    /// origin shape.
    #[test]
    fn e_len_and_e_shape_read_the_descriptor() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[4, 7]);
        let length = e_len(&mut f.ctx(), &Desc::Array(a.clone())).unwrap();
        let term = f.scalar_of(&length).term.as_int().unwrap();
        assert_eq!(f.analysis.terms.constant_value(term), Some(4));
        assert_eq!(f.scalar_of(&length).origin, Origin::Shape);
        let second = e_shape(&mut f.ctx(), &a, 1).unwrap();
        let term = f.scalar_of(&second).term.as_int().unwrap();
        assert_eq!(f.analysis.terms.constant_value(term), Some(7));
    }

    /// §5.2's `E-Arith-Shape`: two shape scalars stay a shape term.
    #[test]
    fn e_arith_on_two_shape_scalars_stays_a_shape_term() {
        let mut f = Fixture::new();
        let (a, b) = (f.literal(6), f.literal(4));
        let key = f.key();
        let sum = e_arith(&mut f.ctx(), ArithOp::Add, &a, &b, &key, "t", None).unwrap();
        let scalar = f.scalar_of(&sum);
        assert_eq!(scalar.origin, Origin::Shape);
        assert_eq!(
            f.analysis
                .terms
                .constant_value(scalar.term.as_int().unwrap()),
            Some(10)
        );
    }

    /// §5.2's `E-Arith-Data`: a data operand makes the result a fresh
    /// variable, and `//` records `t2 ≠ 0` on `Idx`.
    #[test]
    fn e_arith_falls_back_to_a_variable_for_a_data_operand() {
        let mut f = Fixture::new();
        let a = f.literal(6);
        let b = f.data_scalar(Dtype::I64);
        let key = f.key();
        let divided =
            e_arith(&mut f.ctx(), ArithOp::FloorDiv, &a, &b, &key, "t", None).unwrap();
        assert_eq!(f.scalar_of(&divided).origin, Origin::Data);
    }

    /// §5.2: `E-Cmp` on two shape scalars folds; on a data operand it is a
    /// fresh Boolean.
    #[test]
    fn e_cmp_folds_on_shape_scalars_and_is_a_variable_otherwise() {
        let mut f = Fixture::new();
        let (a, b) = (f.literal(2), f.literal(3));
        let key = f.key();
        let less = e_cmp(&mut f.ctx(), CmpOp::Lt, &a, &b, &key, "c");
        match f.scalar_of(&less).term {
            ScalarTerm::Bool(v) => assert!(f.analysis.terms.is_true(v)),
            other => panic!("expected a Boolean term, got {:?}", other),
        }
        let data = f.data_scalar(Dtype::I64);
        let key = f.key();
        let unknown = e_cmp(&mut f.ctx(), CmpOp::Lt, &a, &data, &key, "c");
        assert_eq!(f.scalar_of(&unknown).origin, Origin::Data);
    }

    /// §5.2: `a[i_0, …]` needs one index per axis and records
    /// `⋀_k (-dims_a[k] ≤ t_k < dims_a[k])`.
    #[test]
    fn e_elem_needs_one_index_per_axis() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[4, 3]);
        let i = f.data_scalar(Dtype::I64);
        let key = f.key();
        let element = e_elem(&mut f.ctx(), &a, &[&i, &i], &key, "v").unwrap();
        assert_eq!(f.scalar_of(&element).dtype, Dtype::I64);
        assert_eq!(f.check_rules(), vec!["E-Elem".to_string()]);
        let key = f.key();
        let outcome = e_elem(&mut f.ctx(), &a, &[&i], &key, "v");
        assert!(
            matches!(
                outcome,
                Err(Reject::Premise {
                    rule: Rule::EElem,
                    ..
                })
            ),
            "got {:?}",
            outcome.err()
        );
    }
}
