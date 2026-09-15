//! Boolean indexing and counts: `definition.md` §5.5.
//!
//! `Count` is keyed by the value key of the mask and returns the same
//! variable for the same mask ([`crate::phase0::phi::Phi::count_var`]). That
//! is how $\Phi$ remembers that one mask was used twice. Phase 0 records no
//! relation between the counts of different masks: the count of `~m` is a
//! fresh variable, not `N - c` (§3.4).

use super::helpers::{broadcast_dims, dims_equal, size_term};
use super::Rule;
use super::RuleCtx;
use crate::phase0::desc::{ArrayDesc, Desc, Dtype, DtypeFamily, ValueKey};
use crate::phase0::Reject;

/// `E-Filter-1D` (§5.5): `a[m]` with `a` and `m` one-dimensional.
///
/// It is the rank-1 case of [`e_filter_axis0`], which §5.5 states as the
/// general rule, so the two share one implementation. It exists as a name
/// of its own because §5.5's rule list has both, and the §8.1 example calls
/// it by name; the walker always takes the general rule.
pub fn e_filter_1d(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    m: &ArrayDesc,
    mask: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    e_filter_axis0(ctx, a, m, mask, base)
}

/// `E-Filter-Axis0` (§5.5): `a[m]` with `a` of rank `r` and `m` a
/// one-dimensional Boolean array.
///
/// `new var c = Count(m)`, range `0 ≤ c ≤ dims_m[0]`; output dims are
/// `[c] ++ dims_a[1..]` and `defcond` is `dims_a[0] = dims_m[0]`.
pub fn e_filter_axis0(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    m: &ArrayDesc,
    mask: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    require_boolean_mask(Rule::EFilterAxis0, m)?;
    if m.rank() != 1 {
        return Err(Reject::Premise {
            rule: Rule::EFilterAxis0,
            detail: format!("with a rank-{} mask", m.rank()),
        });
    }
    if a.rank() == 0 {
        return Err(Reject::Premise {
            rule: Rule::EFilterAxis0,
            detail: "on a rank-0 array".to_string(),
        });
    }
    let upper = m.dims[0];
    let c = ctx.phi.count_var(ctx.terms, mask, upper, base);
    let count = ctx.terms.var(c);
    ctx.record_const(Rule::EFilterAxis0.detail("the count"), count);
    let defcond = ctx.terms.eq(a.dims[0], m.dims[0]);
    let mut dims = vec![count];
    dims.extend_from_slice(&a.dims[1..]);
    Ok(ctx.array(Rule::EFilterAxis0, a.dtype, dims, defcond))
}

/// `E-Filter-Same` (§5.5): `a[m]` with `m` of the same shape as `a`.
///
/// The result is one-dimensional of length `c = Count(m)`, with range
/// `0 ≤ c ≤ size(m)`; `defcond` requires `dims_a = dims_m` axis by axis.
pub fn e_filter_same(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    m: &ArrayDesc,
    mask: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    require_boolean_mask(Rule::EFilterSame, m)?;
    if a.rank() != m.rank() {
        return Err(Reject::Premise {
            rule: Rule::EFilterSame,
            detail: format!("with a rank-{} mask on a rank-{} array", m.rank(), a.rank()),
        });
    }
    let upper = size_term(ctx, &m.dims, mask, &format!("sigma_{}", base));
    let c = ctx.phi.count_var(ctx.terms, mask, upper, base);
    let count = ctx.terms.var(c);
    ctx.record_const(Rule::EFilterSame.detail("the count"), count);
    let defcond = dims_equal(ctx.terms, &a.dims, &m.dims);
    Ok(ctx.array(Rule::EFilterSame, a.dtype, vec![count], defcond))
}

/// `E-Nonzero` (§5.5): `nonzero(m)`, `flatnonzero(m)`, the index form of
/// `where(m)`.
///
/// As `E-Filter-Same`, with element dtype `Idx`; the elements have the
/// structural range `[0, size(m))`, which phase 0 notes but does not record,
/// since $\Phi$ does not model array elements. For a mask of rank `r > 1`,
/// `np.nonzero` returns `r` such arrays; the walker wraps `r` of these in a
/// tuple, all sharing the one count variable.
pub fn e_nonzero(
    ctx: &mut RuleCtx,
    m: &ArrayDesc,
    mask: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    require_boolean_mask(Rule::ENonzero, m)?;
    let upper = size_term(ctx, &m.dims, mask, &format!("sigma_{}", base));
    let c = ctx.phi.count_var(ctx.terms, mask, upper, base);
    let count = ctx.terms.var(c);
    ctx.record_const(Rule::ENonzero.detail("the count"), count);
    let tt = ctx.tt();
    Ok(ctx.array(Rule::ENonzero, Dtype::Idx, vec![count], tt))
}

/// `E-Where3` (§5.5): `where(m, a, b)`, an `E-Zip` over the three operands.
pub fn e_where3(
    ctx: &mut RuleCtx,
    m: &Desc,
    a: &Desc,
    b: &Desc,
    out: Dtype,
) -> Result<Desc, Reject> {
    let dims: Vec<Vec<_>> = [m, a, b]
        .iter()
        .map(|d| {
            d.dims().map(|s| s.to_vec()).ok_or_else(|| Reject::Premise {
                rule: Rule::EWhere3,
                detail: format!("on a {}", d.kind_str()),
            })
        })
        .collect::<Result<_, _>>()?;
    let refs: Vec<&[_]> = dims.iter().map(|d| d.as_slice()).collect();
    let (out_dims, defcond) = broadcast_dims(ctx.terms, &refs);
    ctx.prove(Rule::EWhere3.detail("broadcast compatibility"), defcond);
    Ok(ctx.array(Rule::EWhere3, out, out_dims, defcond))
}

/// The premise `dtype(m) = Bool` of §5.5.
///
/// The front end aliases `Boolean` to `Integer` (`coverage.md` §2, §4), so a
/// mask that came in as an input annotation arrives with an integer dtype and
/// cannot be told from an integer array here. The premise is therefore
/// checked at the family level: a float, complex or string mask is a static
/// error, an integer-family one is accepted.
fn require_boolean_mask(rule: Rule, m: &ArrayDesc) -> Result<(), Reject> {
    if m.dtype.family() == DtypeFamily::Integer {
        return Ok(());
    }
    Err(Reject::Premise {
        rule,
        detail: format!("with a {} mask", m.dtype.as_str()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::desc::Desc;
    use crate::phase0::tests::Fixture;

    /// §5.5: `a[m]` on axis 0 gives `[c] ++ dims_a[1..]` with `c = Count(m)`,
    /// and `defcond` is `dims_a[0] = dims_m[0]`.
    #[test]
    fn e_filter_axis0_takes_a_count_and_keeps_the_trailing_axes() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[6, 2]);
        let m = f.input(Dtype::Bool, &[6]);
        let mask = f.key();
        let built = e_filter_axis0(&mut f.ctx(), &a, &m, &mask, "c").unwrap();
        let dims = f.dims_of(&built);
        assert_eq!(dims[0], None, "the leading extent is the count variable");
        assert_eq!(dims[1], Some(2));
        // The extents agree statically, so the condition folds away.
        assert!(f.checks().is_empty(), "got {:?}", f.checks());
    }

    /// §5.5: "`Count` is keyed by the SSA identity of the mask value and
    /// returns the same variable for the same mask".
    #[test]
    fn the_same_mask_gets_the_same_count_variable() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[6]);
        let m = f.input(Dtype::Bool, &[6]);
        let mask = f.key();
        let first = e_filter_axis0(&mut f.ctx(), &a, &m, &mask, "c").unwrap();
        let second = e_filter_axis0(&mut f.ctx(), &a, &m, &mask, "c").unwrap();
        assert_eq!(f.array_of(&first).dims[0], f.array_of(&second).dims[0]);
    }

    /// §5.5's premise `dtype(m) = Bool`, at the family level (§7).
    #[test]
    fn a_float_mask_is_a_premise_violation() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[6]);
        let m = f.input(Dtype::Float, &[6]);
        let mask = f.key();
        let outcome = e_filter_axis0(&mut f.ctx(), &a, &m, &mask, "c");
        assert!(
            matches!(outcome, Err(Reject::Premise { .. })),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.5: a mask of the same shape gives a rank-1 result of length
    /// `Count(m)`, bounded by `size(m)`.
    #[test]
    fn e_filter_same_flattens_to_one_count() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[2, 3]);
        let m = f.input(Dtype::Bool, &[2, 3]);
        let mask = f.key();
        let built = e_filter_same(&mut f.ctx(), &a, &m, &mask, "c").unwrap();
        assert_eq!(f.array_of(&built).rank(), 1);
        assert_eq!(f.dims_of(&built), vec![None]);
    }

    /// §5.5: `nonzero(m)` is `E-Filter-Same` with `Idx` elements, and it
    /// cannot fail.
    #[test]
    fn e_nonzero_is_a_count_of_idx_elements() {
        let mut f = Fixture::new();
        let m = f.input(Dtype::Bool, &[4]);
        let mask = f.key();
        let built = e_nonzero(&mut f.ctx(), &m, &mask, "c").unwrap();
        assert_eq!(f.array_of(&built).dtype, Dtype::Idx);
        assert!(f.checks().is_empty(), "got {:?}", f.checks());
    }

    /// §5.5: `where(m, a, b)` is `E-Zip` over the three operands.
    #[test]
    fn e_where3_broadcasts_the_three_operands() {
        let mut f = Fixture::new();
        let m = Desc::Array(f.input(Dtype::Bool, &[3]));
        let a = Desc::Array(f.input(Dtype::Float, &[1]));
        let b = Desc::Array(f.input(Dtype::Float, &[3]));
        let built = e_where3(&mut f.ctx(), &m, &a, &b, Dtype::Float).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(3)]);
    }

    /// A scalar arm broadcasts as a rank-0 operand, which is what
    /// `np.clip(a, lo, hi)` needs.
    #[test]
    fn e_where3_accepts_a_scalar_arm() {
        let mut f = Fixture::new();
        let m = Desc::Array(f.input(Dtype::Bool, &[3]));
        let a = Desc::Array(f.input(Dtype::Float, &[3]));
        let b = Desc::Scalar(f.literal(0));
        let built = e_where3(&mut f.ctx(), &m, &a, &b, Dtype::Float).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(3)]);
    }
}
