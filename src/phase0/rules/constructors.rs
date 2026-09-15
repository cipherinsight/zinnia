//! Constructors and value-sized lengths: `definition.md` §5.8 and §5.9.

use super::helpers::range_len;
use super::Rule;
use super::{require_int_term, RuleCtx};
use crate::phase0::desc::{ArrayDesc, Desc, Dtype, Origin, ScalarDesc, ValueKey};
use crate::phase0::interval::declared_maximum;
use crate::phase0::phi::VarOrigin;
use crate::phase0::term::{BoolId, Sort, TermId};
use crate::phase0::Reject;

/// The length a value-sized construct gets, with the definedness condition
/// that comes with the declaration (§5.9).
pub struct ValueSizedLength {
    pub length: TermId,
    pub defcond: BoolId,
}

/// `E-ValueSized` (§5.9): `arange(n)`, `zeros(n)`, `repeat(a, k)`, … with
/// `n` or `k` of origin data.
///
/// `t_n` is the length the source asks for before clamping — `n` itself for
/// `arange(n)`, `k * dims_a[axis]` for `repeat` — so the declared maximum
/// always bounds the resulting axis.
///
/// With a declared maximum `M ∈ S` at this operation: a new variable `ℓ`
/// with range `0 ≤ ℓ ≤ M`, the constraint `ℓ = max(0, t_n)`, and `t_n ≤ M`
/// in `defcond` — exceeding `M` is a specified failure, so the equation is a
/// declaration, not a derivation. Without one the program is rejected: this
/// is the only length situation in which phase 0 rejects, because every
/// shape-derived length has a bound.
///
/// `M` comes from the source in two ways. `declared_max` is the one the
/// operation spells out, `size=` or `minlength=`. When the operation spells
/// none, the admitted domain of the scalar is the declaration: a
/// `@requires(lambda N: 0 <= N <= 1024)` bound is a range on `N` in $\Phi$
/// (§3.4), so [`declared_maximum`] reads `M` off `t_n` itself. What is *not*
/// a declaration is the int64 dtype range, which §5.9 calls useless: a bound
/// larger than phase 0 could materialise is reported as missing.
pub fn e_value_sized(
    ctx: &mut RuleCtx,
    t_n: TermId,
    declared_max: Option<i64>,
    value: &ValueKey,
    base: &str,
    construct: &str,
) -> Result<ValueSizedLength, Reject> {
    let maximum = match declared_max {
        Some(m) => m as i128,
        None => declared_maximum(ctx.terms, ctx.phi, t_n, ctx.materialisable).map_err(|why| {
            Reject::ValueSizedWithoutMaximum {
                construct: format!(
                    "{}: its length `{}` {}",
                    construct,
                    ctx.terms.pretty(t_n),
                    why.describe()
                ),
            }
        })?,
    };
    let zero = ctx.terms.constant(0);
    let m = ctx.terms.constant(maximum);
    let ell = ctx.phi.fresh_var(
        ctx.terms,
        base,
        Sort::Idx,
        VarOrigin::ValueSized {
            value: value.clone(),
        },
        Some(zero),
        Some(m),
    );
    let ell_term = ctx.terms.var(ell);
    ctx.record_const(Rule::EValueSized.detail("the declared length"), ell_term);
    let clamped = ctx.terms.max(zero, t_n);
    let declaration = ctx.terms.eq(ell_term, clamped);
    ctx.declare(Rule::EValueSized.detail("declared length"), declaration);
    let defcond = ctx.terms.le(t_n, m);
    Ok(ValueSizedLength {
        length: ell_term,
        defcond,
    })
}

/// `E-Arange` (§5.8): `arange(n)`.
///
/// With `n` of origin shape the length is `max(0, t_n)` and the elements lie
/// in the structural range `[0, t_n)`. With `n` of origin data the length
/// depends on element values, which is [`e_value_sized`].
pub fn e_arange(
    ctx: &mut RuleCtx,
    n: &ScalarDesc,
    declared_max: Option<i64>,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    let t_n = require_int_term(Rule::EArange, n)?;
    let (length, defcond) = if n.origin == Origin::Shape {
        let zero = ctx.terms.constant(0);
        (ctx.terms.max(zero, t_n), ctx.tt())
    } else {
        let sized = e_value_sized(ctx, t_n, declared_max, value, base, "arange")?;
        (sized.length, sized.defcond)
    };
    Ok(ctx.array(Rule::EArange, Dtype::Idx, vec![length], defcond))
}

/// `arange(lo, hi, step)`, the three-argument form, whose length is the
/// [`range_len`] of §5.4 — the same formula as a slice. Endpoints of origin
/// data would make the length depend on element values (§5.9) and reach
/// phase 1 as a useless dtype range, so they are rejected here; a declared
/// maximum makes the one-argument form available instead.
pub fn e_arange_range(
    ctx: &mut RuleCtx,
    lo: &ScalarDesc,
    hi: &ScalarDesc,
    step: i64,
) -> Result<Desc, Reject> {
    for endpoint in [lo, hi] {
        if endpoint.origin != Origin::Shape {
            return Err(Reject::ValueSizedWithoutMaximum {
                construct: "arange(lo, hi, step) with a data-dependent endpoint".to_string(),
            });
        }
    }
    let a = require_int_term(Rule::EArange, lo)?;
    let b = require_int_term(Rule::EArange, hi)?;
    let length = range_len(ctx.terms, a, b, step).ok_or_else(|| Reject::Premise {
        rule: Rule::EArange,
        detail: "with step 0".to_string(),
    })?;
    let tt = ctx.tt();
    Ok(ctx.array(Rule::EArange, Dtype::Idx, vec![length], tt))
}

/// `E-Zeros` (§5.8): `zeros(shape)`, and with it `ones`, `empty`, `full` and
/// the `*_like` family — "as `E-Arange` on each axis".
///
/// An axis of origin data is value-sized; the declared maximum applies to
/// each such axis.
pub fn e_zeros(
    ctx: &mut RuleCtx,
    shape: &[&ScalarDesc],
    dtype: Dtype,
    declared_max: Option<i64>,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    let zero = ctx.terms.constant(0);
    let mut dims = Vec::with_capacity(shape.len());
    let mut conditions = Vec::new();
    for (axis, extent) in shape.iter().enumerate() {
        let t = require_int_term(Rule::EZeros, extent)?;
        if extent.origin == Origin::Shape {
            dims.push(ctx.terms.max(zero, t));
        } else {
            let name = format!("{}_{}", base, axis);
            let sized = e_value_sized(ctx, t, declared_max, value, &name, "zeros")?;
            dims.push(sized.length);
            conditions.push(sized.defcond);
        }
    }
    let defcond = ctx.terms.all(&conditions);
    Ok(ctx.array(Rule::EZeros, dtype, dims, defcond))
}

/// `E-Repeat` (§5.8): `repeat(a, k)`.
///
/// With `k` a constant and an `axis`, `dims[axis] = k * dims_a[axis]`. With
/// `axis = None` NumPy flattens first, so the result is rank 1 of length
/// `k * size(a)`. A `k` of origin data is value-sized (§5.9).
pub fn e_repeat(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    k: &ScalarDesc,
    axis: Option<i64>,
    declared_max: Option<i64>,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    let repeats = match (k.origin, k.term.as_int()) {
        (Origin::Shape, Some(t)) => ctx.terms.constant_value(t),
        _ => None,
    };
    // NumPy flattens first when `axis` is None, so the result is rank 1.
    let (mut dims, target): (Vec<TermId>, usize) = match axis {
        Some(ax) => (
            a.dims.clone(),
            super::helpers::normalise_axis(Rule::ERepeat, ax, a.rank())?,
        ),
        None => {
            let size = super::helpers::size_term(ctx, &a.dims, value, &format!("sigma_{}", base));
            (vec![size], 0)
        }
    };
    let (scaled, defcond) = match repeats {
        Some(n) => (ctx.terms.mul_const(n, dims[target]), ctx.tt()),
        None => {
            // The repeated axis, not the count, is what is value-sized: the
            // declared maximum bounds `k * d`. §3.1 has no product of two
            // non-constant terms, so `d` has to be a literal for that length
            // to be a term at all.
            let extent =
                ctx.terms
                    .constant_value(dims[target])
                    .ok_or_else(|| Reject::Unsupported {
                        construct: "repeat with a data-valued count on a non-constant axis"
                            .to_string(),
                    })?;
            let t_k = require_int_term(Rule::ERepeat, k)?;
            let zero = ctx.terms.constant(0);
            let clamped = ctx.terms.max(zero, t_k);
            let wanted = ctx.terms.mul_const(extent, clamped);
            let sized = e_value_sized(ctx, wanted, declared_max, value, base, "repeat")?;
            (sized.length, sized.defcond)
        }
    };
    dims[target] = scaled;
    Ok(ctx.array(Rule::ERepeat, a.dtype, dims, defcond))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::tests::Fixture;

    /// §5.8: `arange(n)` with `n` of origin shape is `[max(0, t_n)]`.
    #[test]
    fn e_arange_clamps_its_length_at_zero() {
        let mut f = Fixture::new();
        let n = f.literal(5);
        let key = f.key();
        let built = e_arange(&mut f.ctx(), &n, None, &key, "l").unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(5)]);
        let negative = f.literal(-3);
        let key = f.key();
        let empty = e_arange(&mut f.ctx(), &negative, None, &key, "l").unwrap();
        assert_eq!(f.dims_of(&empty), vec![Some(0)]);
    }

    /// §5.8 through §5.4's `range_len`: `arange(lo, hi, step)` has
    /// `max(0, ceil((hi - lo) / step))` elements.
    #[test]
    fn e_arange_range_counts_as_python_does() {
        let mut f = Fixture::new();
        let (lo, hi) = (f.literal(1), f.literal(6));
        let built = e_arange_range(&mut f.ctx(), &lo, &hi, 2).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(3)]);
    }

    /// §5.8: `zeros(shape)` takes one clamped extent per axis.
    #[test]
    fn e_zeros_takes_one_extent_per_axis() {
        let mut f = Fixture::new();
        let (rows, columns) = (f.literal(2), f.literal(3));
        let key = f.key();
        let built = e_zeros(
            &mut f.ctx(),
            &[&rows, &columns],
            Dtype::Float,
            None,
            &key,
            "l",
        )
        .unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(2), Some(3)]);
        assert_eq!(f.array_of(&built).dtype, Dtype::Float);
    }

    /// §5.9: a value-sized extent without a declared maximum rejects.
    #[test]
    fn e_zeros_rejects_a_value_sized_extent_with_no_maximum() {
        let mut f = Fixture::new();
        let n = f.data_scalar(Dtype::I64);
        let key = f.key();
        let outcome = e_zeros(&mut f.ctx(), &[&n], Dtype::Float, None, &key, "l");
        assert!(
            matches!(outcome, Err(Reject::ValueSizedWithoutMaximum { .. })),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.9: with a declared maximum `M` the length is a variable `ℓ` with
    /// `0 ≤ ℓ ≤ M`, and `defcond` demands `t_n ≤ M`.
    #[test]
    fn e_value_sized_declares_a_length_under_its_maximum() {
        let mut f = Fixture::new();
        let n = f.data_scalar(Dtype::I64);
        let t_n = n.term.as_int().unwrap();
        let key = f.key();
        let sized = e_value_sized(&mut f.ctx(), t_n, Some(8), &key, "l", "zeros").unwrap();
        assert_eq!(
            f.analysis.terms.constant_value(sized.length),
            None,
            "the length is the declared variable, not a number"
        );
        assert!(
            !f.analysis.terms.is_true(sized.defcond),
            "the declaration is conditional on `t_n <= M`"
        );
    }

    /// §5.8: `repeat(a, k)` with a constant `k` multiplies the axis.
    #[test]
    fn e_repeat_multiplies_the_axis_by_a_constant_count() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[4]);
        let k = f.literal(3);
        let key = f.key();
        let built = e_repeat(&mut f.ctx(), &a, &k, Some(0), None, &key, "l").unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(12)]);
    }
}
