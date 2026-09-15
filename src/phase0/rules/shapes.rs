//! Shape transformations: `definition.md` §5.6.
//!
//! `E-Transpose`, `E-Moveaxis`, `E-Reshape` (with its `newaxis` / `squeeze`
//! variants, `coverage.md` §5.3), `E-Concat` and `E-Stack`.

use super::helpers::{dims_equal, normalise_axis, normalise_insert_axis, size_term};
use super::Rule;
use super::RuleCtx;
use crate::phase0::desc::{ArrayDesc, Desc, ValueKey};
use crate::phase0::phi::VarOrigin;
use crate::phase0::term::{Sort, TermId};
use crate::phase0::Reject;

/// `E-Transpose` (§5.6): `transpose(a, p)` with `p` a constant permutation.
/// `axes = None` reverses them, as NumPy's default.
pub fn e_transpose(ctx: &mut RuleCtx, a: &ArrayDesc, axes: Option<&[i64]>) -> Result<Desc, Reject> {
    let permutation = match axes {
        Some(p) => super::helpers::permutation(p, a.rank())?,
        None => (0..a.rank()).rev().collect(),
    };
    let dims = permutation.iter().map(|&k| a.dims[k]).collect();
    let tt = ctx.tt();
    Ok(ctx.array(Rule::ETranspose, a.dtype, dims, tt))
}

/// `E-Moveaxis` (§5.6): `moveaxis(a, src, dst)` with constant axes,
/// normalised to a permutation and handed to [`e_transpose`].
pub fn e_moveaxis(ctx: &mut RuleCtx, a: &ArrayDesc, src: i64, dst: i64) -> Result<Desc, Reject> {
    let rank = a.rank();
    let from = normalise_axis(Rule::EMoveaxis, src, rank)?;
    let to = normalise_axis(Rule::EMoveaxis, dst, rank)?;
    let mut order: Vec<i64> = (0..rank as i64).filter(|&k| k != from as i64).collect();
    order.insert(to, from as i64);
    e_transpose(ctx, a, Some(&order))
}

/// One extent of a `reshape` target.
#[derive(Debug, Clone, Copy)]
pub enum ReshapeExtent {
    /// A literal extent.
    Const(i64),
    /// The single `-1` whose extent is inferred.
    Infer,
    /// A shape-derived extent, for example `a.shape[0]`.
    Term(TermId),
}

/// `E-Reshape` (§5.6): `reshape(a, shape)`, entries constant or exactly one
/// `-1`.
///
/// Premise: `size(a)` must be a term, that is at most one axis of `a` is
/// non-constant; otherwise the rule rejects (§3.4, end, covers the
/// alternative for rules that can use a $\sigma$ variable, but a reshape
/// needs the size itself). With a `-1`, the product `p` of the other entries
/// must be a constant, so a target with a non-constant extent next to `-1`
/// rejects — `reshape(a, (n, -1))` with `n` of origin data would make the
/// inferred axis `size(a) div n`, a division by a variable (§5.6, last
/// paragraph).
pub fn e_reshape(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    shape: &[ReshapeExtent],
    value: &ValueKey,
) -> Result<Desc, Reject> {
    if shape
        .iter()
        .filter(|e| matches!(e, ReshapeExtent::Infer))
        .count()
        > 1
    {
        return Err(Reject::ReshapeNotSupported {
            detail: "more than one -1 in the target shape".to_string(),
        });
    }
    let infer_at = shape.iter().position(|e| matches!(e, ReshapeExtent::Infer));
    let size = size_term(ctx, &a.dims, value, "sigma_reshape_in");
    let Some(axis) = infer_at else {
        return reshape_exact(ctx, a, shape, size, value);
    };
    match constant_product(shape, axis) {
        // §5.6's own case: the inferred axis is `size(a) div p`.
        Some(p) if p > 0 => reshape_inferred(ctx, a, shape, axis, size, p),
        // A `-1` next to a non-constant extent, or a product that is not
        // positive: the inferred axis becomes a variable and the product
        // fact of §3.4 carries the nonlinear part; see "Additions" in
        // `implementation.md`.
        _ => reshape_declared(ctx, a, shape, axis, size, value),
    }
}

/// The product of every target extent but `axis`, when all of them are
/// literals. `None` as soon as one is a term: that is the predicate the
/// division case and the declared case differ on.
fn constant_product(shape: &[ReshapeExtent], axis: usize) -> Option<i128> {
    let mut product: i128 = 1;
    for (k, extent) in shape.iter().enumerate() {
        if k == axis {
            continue;
        }
        match *extent {
            ReshapeExtent::Const(n) => product *= n as i128,
            _ => return None,
        }
    }
    Some(product)
}

/// The extent terms of a reshape target, with `inferred` standing for the
/// `-1` axis. The three paths differ in what `inferred` is and in the
/// `defcond` they record, not in how the vector is built.
fn target_dims(
    ctx: &mut RuleCtx,
    shape: &[ReshapeExtent],
    inferred: Option<TermId>,
) -> Result<Vec<TermId>, Reject> {
    shape
        .iter()
        .map(|extent| match *extent {
            ReshapeExtent::Const(n) => Ok(ctx.terms.constant(n as i128)),
            ReshapeExtent::Term(t) => Ok(t),
            ReshapeExtent::Infer => inferred.ok_or_else(|| Reject::ReshapeNotSupported {
                detail: "a -1 with no inferred extent".to_string(),
            }),
        })
        .collect()
}

/// The no-`-1` case: `defcond = (size(a) = ∏ shape)`.
fn reshape_exact(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    shape: &[ReshapeExtent],
    size: TermId,
    value: &ValueKey,
) -> Result<Desc, Reject> {
    let dims = target_dims(ctx, shape, None)?;
    let target = size_term(ctx, &dims, value, "sigma_reshape_out");
    let defcond = ctx.terms.eq(size, target);
    ctx.prove(Rule::EReshape.detail("the sizes agree"), defcond);
    Ok(ctx.array(Rule::EReshape, a.dtype, dims, defcond))
}

/// The `-1` case when the other extents are not all constant.
///
/// The inferred axis would be `size(a) div p` with `p` a runtime value, which
/// §3.1 has no term for. Instead a variable `ℓ` with `0 ≤ ℓ ≤ size(a)` stands
/// for it and `defcond` is `p > 0 ∧ size(a) = ∏ target`, the product fact of
/// §3.4 carrying the nonlinear part. This is the same device §3.4 prescribes
/// for a size whose factors are not all constant.
///
/// `p > 0` is §5.6's own condition on the constant case, and it is what makes
/// the check sound here: NumPy cannot infer `-1` when the product of the
/// other extents is `0`, and without the conjunct `size(a) = 0 = ∏ target`
/// would satisfy `defcond` on a run that raises (rule defect 2 of
/// `implementation.md`).
fn reshape_declared(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    shape: &[ReshapeExtent],
    axis: usize,
    size: TermId,
    value: &ValueKey,
) -> Result<Desc, Reject> {
    let zero = ctx.terms.constant(0);
    let ell = ctx.phi.fresh_var(
        ctx.terms,
        "l_reshape",
        Sort::Idx,
        VarOrigin::ValueSized {
            value: value.clone(),
        },
        Some(zero),
        Some(size),
    );
    let inferred = ctx.terms.var(ell);
    ctx.record_const(Rule::EReshape.detail("the inferred extent"), inferred);
    let dims = target_dims(ctx, shape, Some(inferred))?;
    let others: Vec<TermId> = dims
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != axis)
        .map(|(_, &d)| d)
        .collect();
    let p = size_term(ctx, &others, value, "sigma_reshape_p");
    let positive = ctx.terms.lt(zero, p);
    let target = size_term(ctx, &dims, value, "sigma_reshape_out");
    let sizes_agree = ctx.terms.eq(size, target);
    ctx.prove(Rule::EReshape.detail("the sizes agree"), sizes_agree);
    let defcond = ctx.terms.and(positive, sizes_agree);
    Ok(ctx.array(Rule::EReshape, a.dtype, dims, defcond))
}

/// The `-1` case of §5.6, with `p` the constant product of the other
/// extents: the inferred axis is `size(a) div p` and `defcond` is
/// `size(a) mod p = 0`. `p > 0` is the caller's precondition, so §5.6's
/// other conjunct holds by construction.
fn reshape_inferred(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    shape: &[ReshapeExtent],
    axis: usize,
    size: TermId,
    p: i128,
) -> Result<Desc, Reject> {
    // `p > 0`, so neither division is by zero; the guard is here because
    // the arena reports it rather than panicking.
    let not_divisible = || Reject::ReshapeNotSupported {
        detail: "the product of the extents next to -1 is zero".to_string(),
    };
    let remainder = ctx.terms.mod_const(size, p).ok_or_else(not_divisible)?;
    let zero = ctx.terms.constant(0);
    let defcond = ctx.terms.eq(remainder, zero);
    let inferred = ctx.terms.div_const(size, p).ok_or_else(not_divisible)?;
    ctx.record_const(Rule::EReshape.detail("the inferred extent"), inferred);
    let mut dims = target_dims(ctx, shape, Some(inferred))?;
    dims[axis] = inferred;
    Ok(ctx.array(Rule::EReshape, a.dtype, dims, defcond))
}

/// `E-NewAxis` (`coverage.md` §5.3, a reshape variant): `expand_dims(a, k)`
/// and `a[..., None]`. A constant 1 is inserted at `axis`; the rule cannot
/// fail.
pub fn e_expand_dims(ctx: &mut RuleCtx, a: &ArrayDesc, axis: i64) -> Result<Desc, Reject> {
    let at = normalise_insert_axis(Rule::ENewAxis, axis, a.rank())?;
    let one = ctx.terms.constant(1);
    let mut dims = a.dims.clone();
    dims.insert(at, one);
    let tt = ctx.tt();
    Ok(ctx.array(Rule::ENewAxis, a.dtype, dims, tt))
}

/// `squeeze(a, axis)`, the inverse of [`e_expand_dims`]. `defcond` is
/// `dims_a[axis] = 1`; with `axis = None` every statically-1 axis is dropped,
/// which is NumPy's behaviour on the shapes phase 0 can see.
pub fn e_squeeze(ctx: &mut RuleCtx, a: &ArrayDesc, axis: Option<i64>) -> Result<Desc, Reject> {
    let one = ctx.terms.constant(1);
    let (dims, defcond) = match axis {
        Some(k) => {
            let at = normalise_axis(Rule::ESqueeze, k, a.rank())?;
            let defcond = ctx.terms.eq(a.dims[at], one);
            let dims = a
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != at)
                .map(|(_, &d)| d)
                .collect();
            (dims, defcond)
        }
        None => {
            let dims = a.dims.iter().copied().filter(|&d| d != one).collect();
            (dims, ctx.tt())
        }
    };
    Ok(ctx.array(Rule::ESqueeze, a.dtype, dims, defcond))
}

/// `E-Concat` (§5.6): `concatenate([a, b, ...], axis=k)`.
///
/// `dims[k]` is the sum of the inputs' extents on that axis; the other axes
/// come from the first input and `defcond` is `⋀_{j≠k} (dims_a[j] =
/// dims_b[j])`.
pub fn e_concat(ctx: &mut RuleCtx, parts: &[&ArrayDesc], axis: i64) -> Result<Desc, Reject> {
    let first = require_same_rank(Rule::EConcat, parts)?;
    let k = normalise_axis(Rule::EConcat, axis, first.rank())?;
    let mut dims = first.dims.clone();
    let mut conditions = Vec::new();
    for part in &parts[1..] {
        dims[k] = ctx.terms.add(dims[k], part.dims[k]);
        for (j, (&p, &q)) in first.dims.iter().zip(part.dims.iter()).enumerate() {
            if j != k {
                conditions.push(ctx.terms.eq(p, q));
            }
        }
    }
    let defcond = ctx.terms.all(&conditions);
    Ok(ctx.array(Rule::EConcat, first.dtype, dims, defcond))
}

/// `E-Stack` (§5.6): `stack([a, b, ...], axis=k)`.
///
/// The constant number of parts is inserted at position `k` and `defcond` is
/// `dims_a = dims_b` for every pair.
pub fn e_stack(ctx: &mut RuleCtx, parts: &[&ArrayDesc], axis: i64) -> Result<Desc, Reject> {
    let first = require_same_rank(Rule::EStack, parts)?;
    let at = normalise_insert_axis(Rule::EStack, axis, first.rank())?;
    let mut conditions = Vec::new();
    for part in &parts[1..] {
        conditions.push(dims_equal(ctx.terms, &first.dims, &part.dims));
    }
    let defcond = ctx.terms.all(&conditions);
    let count = ctx.terms.constant(parts.len() as i128);
    let mut dims = first.dims.clone();
    dims.insert(at, count);
    Ok(ctx.array(Rule::EStack, first.dtype, dims, defcond))
}

/// The premise shared by `E-Concat` and `E-Stack`: a non-empty list of
/// arrays of equal rank. Unequal ranks are a static error.
fn require_same_rank<'d>(rule: Rule, parts: &[&'d ArrayDesc]) -> Result<&'d ArrayDesc, Reject> {
    let first = parts.first().ok_or_else(|| Reject::Premise {
        rule,
        detail: "of an empty list".to_string(),
    })?;
    if parts.iter().any(|p| p.rank() != first.rank()) {
        return Err(Reject::Premise {
            rule,
            detail: "of arrays of different rank".to_string(),
        });
    }
    Ok(first)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::desc::Dtype;
    use crate::phase0::tests::Fixture;

    /// §5.6: `transpose(a, p)` permutes the extents; the default reverses.
    #[test]
    fn e_transpose_permutes_the_extents() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[2, 3, 4]);
        let reversed = e_transpose(&mut f.ctx(), &a, None).unwrap();
        assert_eq!(f.dims_of(&reversed), vec![Some(4), Some(3), Some(2)]);
        let chosen = e_transpose(&mut f.ctx(), &a, Some(&[1, 0, 2])).unwrap();
        assert_eq!(f.dims_of(&chosen), vec![Some(3), Some(2), Some(4)]);
    }

    /// A permutation that is not one is a premise violation (§5.6).
    #[test]
    fn e_transpose_rejects_a_repeated_axis() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[2, 3]);
        let outcome = e_transpose(&mut f.ctx(), &a, Some(&[0, 0]));
        assert!(
            matches!(
                outcome,
                Err(Reject::Premise {
                    rule: Rule::ETranspose,
                    ..
                })
            ),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.6: `moveaxis(a, src, dst)` normalises to a permutation (rule B.1).
    #[test]
    fn e_moveaxis_moves_one_axis() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[2, 3, 4]);
        let moved = e_moveaxis(&mut f.ctx(), &a, 0, 2).unwrap();
        assert_eq!(f.dims_of(&moved), vec![Some(3), Some(4), Some(2)]);
    }

    /// §5.6, the no-`-1` case: the target is the output, and `defcond` is
    /// `size(a) = ∏ shape`.
    #[test]
    fn e_reshape_without_an_inferred_axis_takes_the_target() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[6]);
        let key = f.key();
        let built = e_reshape(
            &mut f.ctx(),
            &a,
            &[ReshapeExtent::Const(2), ReshapeExtent::Const(3)],
            &key,
        )
        .unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(2), Some(3)]);
        assert!(f.checks().is_empty(), "6 = 2 * 3 folds away");
    }

    /// §5.6, the `-1` case with a constant product: the axis is
    /// `size(a) div p`.
    #[test]
    fn e_reshape_divides_when_the_other_extents_are_constant() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[8]);
        let key = f.key();
        let built = e_reshape(
            &mut f.ctx(),
            &a,
            &[ReshapeExtent::Infer, ReshapeExtent::Const(2)],
            &key,
        )
        .unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(4), Some(2)]);
    }

    /// `implementation.md` §8: a `-1` beside a non-constant extent takes the
    /// declared-variable path instead of a division.
    #[test]
    fn e_reshape_declares_a_variable_beside_a_non_constant_extent() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[8]);
        let n = f.symbolic_input(Dtype::I64, "n", 8);
        let key = f.key();
        let built = e_reshape(
            &mut f.ctx(),
            &a,
            &[ReshapeExtent::Infer, ReshapeExtent::Term(n.dims[0])],
            &key,
        )
        .unwrap();
        let dims = f.dims_of(&built);
        assert_eq!(dims, vec![None, None], "both extents are terms");
        assert_eq!(f.check_rules(), vec!["E-Reshape".to_string()]);
    }

    /// Two `-1`s have no rule at all (§5.6: "exactly one `-1`").
    #[test]
    fn e_reshape_rejects_two_inferred_axes() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[8]);
        let key = f.key();
        let outcome = e_reshape(
            &mut f.ctx(),
            &a,
            &[ReshapeExtent::Infer, ReshapeExtent::Infer],
            &key,
        );
        assert!(
            matches!(outcome, Err(Reject::ReshapeNotSupported { .. })),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.6: `concatenate` adds along the axis and demands the others agree.
    #[test]
    fn e_concat_adds_the_axis_and_checks_the_others() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[2, 3]);
        let b = f.input(Dtype::I64, &[4, 3]);
        let built = e_concat(&mut f.ctx(), &[&a, &b], 0).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(6), Some(3)]);
        assert!(f.checks().is_empty(), "the other axis agrees statically");
    }

    /// Unequal ranks are a premise violation of the same rule.
    #[test]
    fn e_concat_rejects_operands_of_different_rank() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[2, 3]);
        let b = f.input(Dtype::I64, &[4]);
        let outcome = e_concat(&mut f.ctx(), &[&a, &b], 0);
        assert!(
            matches!(
                outcome,
                Err(Reject::Premise {
                    rule: Rule::EConcat,
                    ..
                })
            ),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.6: `stack` inserts the operand count at the axis.
    #[test]
    fn e_stack_inserts_the_operand_count() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[3]);
        let b = f.input(Dtype::I64, &[3]);
        let built = e_stack(&mut f.ctx(), &[&a, &b], 0).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(2), Some(3)]);
    }

    /// `implementation.md` §8: `expand_dims` inserts a 1, `squeeze` drops
    /// one, with `defcond = (dims_a[k] = 1)`.
    #[test]
    fn e_expand_dims_and_e_squeeze_are_inverse() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::I64, &[3]);
        let widened = e_expand_dims(&mut f.ctx(), &a, 0).unwrap();
        assert_eq!(f.dims_of(&widened), vec![Some(1), Some(3)]);
        let wide = f.array_of(&widened);
        let narrowed = e_squeeze(&mut f.ctx(), &wide, Some(0)).unwrap();
        assert_eq!(f.dims_of(&narrowed), vec![Some(3)]);
        assert!(f.checks().is_empty(), "the extent is the literal 1");
    }
}
