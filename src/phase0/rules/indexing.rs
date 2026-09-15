//! Indexing and slicing: `definition.md` §5.4.
//!
//! `E-Index`, `E-Slice` and `E-Take`. The slice length divides by a
//! constant, so it stays in the term language; a runtime step is out of
//! scope and rejected (§5.4).

use super::helpers::{in_bounds, norm, range_len};
use super::scalars::e_elem;
use super::Rule;
use super::{require_int_term, RuleCtx};
use crate::phase0::desc::{ArrayDesc, Desc, ScalarDesc, ValueKey};
use crate::phase0::term::TermId;
use crate::phase0::Reject;

/// `E-Index` (§5.4): `a[i]`, an integer scalar index on `axis`.
///
/// The rank drops by one and the axis is removed;
/// `defcond = (-dims_a[axis] ≤ t_i < dims_a[axis])`. Indexing the only axis
/// of a rank-1 array yields a scalar, which is `E-Elem` (§5.2) with one
/// index, so that case delegates rather than duplicating the rule.
pub fn e_index(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    axis: usize,
    index: &ScalarDesc,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    if axis >= a.rank() {
        return Err(Reject::Premise {
            rule: Rule::EIndex,
            detail: format!("on axis {} of a rank-{} array", axis, a.rank()),
        });
    }
    if a.rank() == 1 {
        return e_elem(ctx, a, &[index], value, base);
    }
    let t = require_int_term(Rule::EIndex, index)?;
    let defcond = in_bounds(ctx.terms, t, a.dims[axis]);
    let dims: Vec<TermId> = a
        .dims
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != axis)
        .map(|(_, &d)| d)
        .collect();
    Ok(ctx.array(Rule::EIndex, a.dtype, dims, defcond))
}

/// `E-Slice` (§5.4): `a[lo:hi:step]` on `axis`, `step` a non-zero constant.
///
/// An omitted `lo` is `0` for a positive step and `d - 1` for a negative
/// one; an omitted `hi` is `d` or `-d-1`, per Python. Both endpoints go
/// through [`norm`]; the length is [`range_len`]. The rule cannot fail, so
/// `defcond` is true.
pub fn e_slice(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    axis: usize,
    lo: Option<&ScalarDesc>,
    hi: Option<&ScalarDesc>,
    step: i64,
) -> Result<Desc, Reject> {
    if axis >= a.rank() {
        return Err(Reject::Premise {
            rule: Rule::ESlice,
            detail: format!("on axis {} of a rank-{} array", axis, a.rank()),
        });
    }
    let len = slice_extent(ctx, a.dims[axis], lo, hi, step)?;
    let mut dims = a.dims.clone();
    dims[axis] = len;
    let tt = ctx.tt();
    Ok(ctx.array(Rule::ESlice, a.dtype, dims, tt))
}

/// The extent `a[lo:hi:step]` leaves on an axis of extent `d` (§5.4).
///
/// This is the length half of [`e_slice`], separated because the multi-index
/// form of a subscript needs the extent without the descriptor
/// (`walk::subscript`).
pub fn slice_extent(
    ctx: &mut RuleCtx,
    d: TermId,
    lo: Option<&ScalarDesc>,
    hi: Option<&ScalarDesc>,
    step: i64,
) -> Result<TermId, Reject> {
    if step == 0 {
        return Err(Reject::Premise {
            rule: Rule::ESlice,
            detail: "with step 0".to_string(),
        });
    }
    let negative = step < 0;
    let one = ctx.terms.constant(1);
    let default_lo = if negative {
        ctx.terms.sub(d, one)
    } else {
        ctx.terms.constant(0)
    };
    let default_hi = if negative {
        // Python's "one before the start": -d-1.
        let neg_d = ctx.terms.mul_const(-1, d);
        ctx.terms.sub(neg_d, one)
    } else {
        d
    };
    let lo_raw = endpoint(lo, default_lo)?;
    let hi_raw = endpoint(hi, default_hi)?;
    let l = norm(ctx.terms, lo_raw, d, negative);
    let u = norm(ctx.terms, hi_raw, d, negative);
    range_len(ctx.terms, l, u, step).ok_or_else(|| Reject::Premise {
        rule: Rule::ESlice,
        detail: "with step 0".to_string(),
    })
}

fn endpoint(given: Option<&ScalarDesc>, default: TermId) -> Result<TermId, Reject> {
    match given {
        None => Ok(default),
        Some(s) => require_int_term(Rule::ESlice, s),
    }
}

/// `E-Take` (§5.4): `take(a, I, axis=k)` with `I` an integer array.
///
/// `defcond = inrange(I, dims_a[k])`, which is a fresh Boolean variable
/// $\beta_I$ standing for "every element of `I` lies in `[-d, d)`". $\Phi$
/// does not model the elements of `I`, so $\beta_I$ is unconstrained and
/// CANFAIL answers "may fail". This is the boundary of $\Phi$: soundness is
/// kept ($\Phi$ excludes no real state) and precision is lost.
pub fn e_take(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    indices: &ArrayDesc,
    axis: usize,
    index_value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    if axis >= a.rank() {
        return Err(Reject::Premise {
            rule: Rule::ETake,
            detail: format!("on axis {} of a rank-{} array", axis, a.rank()),
        });
    }
    let defcond = ctx.definedness(base, index_value);
    let mut dims: Vec<TermId> = a.dims[..axis].to_vec();
    dims.extend_from_slice(&indices.dims);
    dims.extend_from_slice(&a.dims[axis + 1..]);
    Ok(ctx.array(Rule::ETake, a.dtype, dims, defcond))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::desc::Dtype;
    use crate::phase0::tests::Fixture;

    /// §5.4: `a[i]` drops axis 0 and records `-d ≤ t_i < d`.
    #[test]
    fn e_index_drops_the_axis_and_records_the_bounds_check() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[4, 3]);
        let i = f.literal(1);
        let key = f.key();
        let built = e_index(&mut f.ctx(), &a, 0, &i, &key, "v").unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(3)]);
        // A constant index inside the extent folds the condition away.
        assert!(f.checks().is_empty(), "got {:?}", f.checks());
    }

    /// The same rule with a runtime index keeps the condition. A rank-1
    /// operand would delegate to `E-Elem` (§5.2), so the test indexes a
    /// rank-2 one.
    #[test]
    fn e_index_with_a_runtime_index_records_a_check() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[4, 3]);
        let i = f.data_scalar(Dtype::I64);
        let key = f.key();
        e_index(&mut f.ctx(), &a, 0, &i, &key, "v").unwrap();
        assert_eq!(f.check_rules(), vec!["E-Index".to_string()]);
    }

    /// §5.4's premise: the axis must exist.
    #[test]
    fn e_index_on_a_missing_axis_is_a_premise_violation() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[4]);
        let i = f.literal(0);
        let key = f.key();
        let outcome = e_index(&mut f.ctx(), &a, 1, &i, &key, "v");
        assert!(
            matches!(
                outcome,
                Err(Reject::Premise {
                    rule: Rule::EIndex,
                    ..
                })
            ),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.4: a slice keeps the rank and cannot fail.
    #[test]
    fn e_slice_keeps_the_rank_and_records_no_check() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[7, 2]);
        let (lo, hi) = (f.literal(1), f.literal(6));
        let built = e_slice(&mut f.ctx(), &a, 0, Some(&lo), Some(&hi), 2).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(3), Some(2)]);
        assert!(f.checks().is_empty(), "a slice cannot fail");
    }

    /// §5.4: a step of zero is a premise violation, not a length of zero.
    #[test]
    fn e_slice_with_step_zero_is_a_premise_violation() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[7]);
        let outcome = e_slice(&mut f.ctx(), &a, 0, None, None, 0);
        assert!(
            matches!(
                outcome,
                Err(Reject::Premise {
                    rule: Rule::ESlice,
                    ..
                })
            ),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.4: `take(a, I, axis=k)` replaces axis `k` by `dims_I`, and its
    /// `inrange` is an unconstrained Boolean, so the check stands.
    #[test]
    fn e_take_replaces_the_axis_with_the_index_arrays_dims() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::Float, &[5, 2]);
        let indices = f.input(Dtype::I64, &[3]);
        let key = f.key();
        let built = e_take(&mut f.ctx(), &a, &indices, 0, &key, "in").unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(3), Some(2)]);
        assert_eq!(f.check_rules(), vec!["E-Take".to_string()]);
    }
}
