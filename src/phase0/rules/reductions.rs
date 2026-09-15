//! Reductions, matmul and sorting: `definition.md` §5.7, plus
//! `E-Reduce-Scalar` of §5.2, which shares [`ReduceKind`] with
//! `E-Reduce-Axis`.

use super::helpers::{broadcast_dims, normalise_axis, size_term};
use super::Rule;
use super::{RuleCtx, VarRange};
use crate::phase0::desc::{ArrayDesc, Desc, Dtype, ValueKey};
use crate::phase0::phi::VarOrigin;
use crate::phase0::term::{BoolId, Sort, TermId};
use crate::phase0::Reject;

/// The reduction operators of `coverage.md` §5.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Prod,
    Min,
    Max,
    ArgMin,
    ArgMax,
    Any,
    All,
    Mean,
    Var,
    Std,
    Norm,
}

impl ReduceKind {
    /// Whether the reduction is undefined on an empty axis, which is
    /// §5.7's `defcond = (dims_a[k] > 0 for max, min, argmax)`.
    pub fn needs_nonempty(self) -> bool {
        matches!(
            self,
            ReduceKind::Min | ReduceKind::Max | ReduceKind::ArgMin | ReduceKind::ArgMax
        )
    }

    /// Whether the result is a position rather than a value, which gives it
    /// the structural range `[0, size)` of §5.2.
    pub fn is_index(self) -> bool {
        matches!(self, ReduceKind::ArgMin | ReduceKind::ArgMax)
    }

    /// The result dtype, following NumPy: a position is an index, a
    /// predicate is Boolean, a statistic is a float, and a sum keeps the
    /// element dtype except that Booleans sum to integers.
    pub fn result_dtype(self, input: Dtype) -> Dtype {
        match self {
            ReduceKind::ArgMin | ReduceKind::ArgMax => Dtype::Idx,
            ReduceKind::Any | ReduceKind::All => Dtype::Bool,
            ReduceKind::Mean | ReduceKind::Var | ReduceKind::Std | ReduceKind::Norm => Dtype::Float,
            ReduceKind::Sum | ReduceKind::Prod if input == Dtype::Bool => Dtype::I64,
            _ => input,
        }
    }
}

/// `E-Reduce-Scalar` (§5.2): `sum(a)`, `max(a)`, `argmax(a)`, … over all
/// elements.
///
/// The result is a fresh data variable. `sum` gets the dtype range;
/// `argmax` gets the structural range `0 ≤ v < size(a)`. `defcond` is
/// `size(a) > 0` for `max`, `min` and `argmax`, true otherwise.
pub fn e_reduce_scalar(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    kind: ReduceKind,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    let size = size_term(ctx, &a.dims, value, &format!("sigma_{}", base));
    let dtype = kind.result_dtype(a.dtype);
    let range = if kind.is_index() {
        let zero = ctx.terms.constant(0);
        VarRange::structural(ctx.terms, zero, size)
    } else {
        VarRange::of_dtype(ctx.terms, dtype)
    };
    let defcond = if kind.needs_nonempty() {
        let zero = ctx.terms.constant(0);
        ctx.terms.lt(zero, size)
    } else {
        ctx.tt()
    };
    Ok(ctx.data_scalar(Rule::EReduceScalar, dtype, base, value, range, defcond))
}

/// `E-Reduce-Axis` (§5.7): `sum(a, axis=k)`, `max(a, axis=k)`, …
///
/// The reduced axes are removed, or set to 1 with `keepdims`. `defcond` is
/// `dims_a[k] > 0` for `max`, `min` and `argmax`.
pub fn e_reduce_axis(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    kind: ReduceKind,
    axes: &[i64],
    keepdims: bool,
) -> Result<Desc, Reject> {
    let mut reduced = Vec::with_capacity(axes.len());
    for &axis in axes {
        reduced.push(normalise_axis(Rule::EReduceAxis, axis, a.rank())?);
    }
    let zero = ctx.terms.constant(0);
    let mut conditions = Vec::new();
    if kind.needs_nonempty() {
        for &k in &reduced {
            conditions.push(ctx.terms.lt(zero, a.dims[k]));
        }
    }
    let defcond = ctx.terms.all(&conditions);
    let one = ctx.terms.constant(1);
    let mut dims = Vec::with_capacity(a.rank());
    for (k, &d) in a.dims.iter().enumerate() {
        match (reduced.contains(&k), keepdims) {
            (true, false) => {}
            (true, true) => dims.push(one),
            (false, _) => dims.push(d),
        }
    }
    let dtype = kind.result_dtype(a.dtype);
    Ok(ctx.array(Rule::EReduceAxis, dtype, dims, defcond))
}

/// `E-Matmul` (§5.7): `a @ b` with `a: α ++ [M, K]` and `b: β ++ [K', N]`.
///
/// Premise: rank ≥ 1. A one-dimensional operand is promoted to `[1, K]` on
/// the left or `[K, 1]` on the right, and the promoted axis is removed from
/// the result. The batch axes broadcast; `defcond` is `(K = K')` together
/// with the broadcasting condition on those axes.
pub fn e_matmul(ctx: &mut RuleCtx, a: &ArrayDesc, b: &ArrayDesc) -> Result<Desc, Reject> {
    if a.rank() == 0 || b.rank() == 0 {
        return Err(Reject::Premise {
            rule: Rule::EMatmul,
            detail: "on a rank-0 operand".to_string(),
        });
    }
    let one = ctx.terms.constant(1);
    let promoted_left = a.rank() == 1;
    let promoted_right = b.rank() == 1;
    let left: Vec<TermId> = if promoted_left {
        vec![one, a.dims[0]]
    } else {
        a.dims.clone()
    };
    let right: Vec<TermId> = if promoted_right {
        vec![b.dims[0], one]
    } else {
        b.dims.clone()
    };
    let (m, k) = (left[left.len() - 2], left[left.len() - 1]);
    let (k2, n) = (right[right.len() - 2], right[right.len() - 1]);
    let (batch, broadcast_ok) = broadcast_dims(
        ctx.terms,
        &[&left[..left.len() - 2], &right[..right.len() - 2]],
    );
    let contract_ok = ctx.terms.eq(k, k2);
    ctx.prove(
        Rule::EMatmul.detail("the inner dimensions agree"),
        contract_ok,
    );
    ctx.prove(
        Rule::EMatmul.detail("broadcast compatibility of the batch axes"),
        broadcast_ok,
    );
    let defcond = ctx.terms.and(contract_ok, broadcast_ok);
    let mut dims = batch;
    if !promoted_left {
        dims.push(m);
    }
    if !promoted_right {
        dims.push(n);
    }
    let dtype = Dtype::promote(a.dtype, b.dtype).unwrap_or(a.dtype);
    Ok(ctx.array(Rule::EMatmul, dtype, dims, defcond))
}

/// `E-Sort` (§5.7): `sort(a)` and `argsort(a)`.
///
/// The dimensions are unchanged and the rule cannot fail. `argsort`
/// (`index_result`) has element dtype `Idx`; its elements lie in
/// `[0, dims_a[0])`, a structural range on elements that $\Phi$ does not
/// record because it does not model elements.
pub fn e_sort(ctx: &mut RuleCtx, a: &ArrayDesc, index_result: bool) -> Result<Desc, Reject> {
    let dtype = if index_result { Dtype::Idx } else { a.dtype };
    let dims = a.dims.clone();
    let tt = ctx.tt();
    Ok(ctx.array(Rule::ESort, dtype, dims, tt))
}

/// `E-Cumsum` (§5.7): `cumsum(a)`, `cumprod(a)`. Dimensions unchanged.
pub fn e_cumsum(ctx: &mut RuleCtx, a: &ArrayDesc) -> Result<Desc, Reject> {
    let dims = a.dims.clone();
    let tt = ctx.tt();
    Ok(ctx.array(Rule::ECumsum, a.dtype, dims, tt))
}

/// `E-Reverse` (§5.7): `flip`, `flipud`, `fliplr`, `a[::-1]` on a whole
/// axis. Dimensions unchanged.
pub fn e_reverse(ctx: &mut RuleCtx, a: &ArrayDesc) -> Result<Desc, Reject> {
    let dims = a.dims.clone();
    let tt = ctx.tt();
    Ok(ctx.array(Rule::EReverse, a.dtype, dims, tt))
}

/// `E-Unique` (§5.7): `unique(a)`.
///
/// `new var c'`, range `0 ≤ c' ≤ size(a)`, plus `size(a) > 0 ⇒ c' ≥ 1`. The
/// count of `unique` is not the count of any mask, so it is an independent
/// variable with only the trivial range — the precision limit of $\Phi$ for
/// this operator.
pub fn e_unique(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    let size = size_term(ctx, &a.dims, value, &format!("sigma_{}", base));
    let zero = ctx.terms.constant(0);
    let c = ctx.phi.fresh_var(
        ctx.terms,
        base,
        Sort::Idx,
        VarOrigin::UniqueCount {
            value: value.clone(),
        },
        Some(zero),
        Some(size),
    );
    let c_term = ctx.terms.var(c);
    ctx.record_const(Rule::EUnique.detail("the count"), c_term);
    let nonempty: BoolId = ctx.terms.lt(zero, size);
    let one = ctx.terms.constant(1);
    let at_least_one = ctx.terms.le(one, c_term);
    let fact = ctx.terms.implies(nonempty, at_least_one);
    ctx.declare(
        Rule::EUnique.detail("a non-empty input has a non-empty result"),
        fact,
    );
    let dims = vec![c_term];
    let tt = ctx.tt();
    Ok(ctx.array(Rule::EUnique, a.dtype, dims, tt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::desc::Dtype;
    use crate::phase0::tests::Fixture;

    /// §5.2: `sum(a)` is a data scalar and cannot fail; `max(a)` demands
    /// `size(a) > 0`.
    #[test]
    fn e_reduce_scalar_records_a_condition_only_where_emptiness_matters() {
        let mut f = Fixture::new();
        let a = f.symbolic_input(Dtype::Float, "n", 16);
        let key = f.key();
        e_reduce_scalar(&mut f.ctx(), &a, ReduceKind::Sum, &key, "s").unwrap();
        assert!(f.checks().is_empty(), "a sum over an empty array is 0");
        let key = f.key();
        e_reduce_scalar(&mut f.ctx(), &a, ReduceKind::Max, &key, "m").unwrap();
        assert_eq!(f.check_rules(), vec!["E-Reduce-Scalar".to_string()]);
    }

    /// §5.2: `argmax` has the structural range `[0, size(a))` (§3.4).
    #[test]
    fn e_reduce_scalar_gives_argmax_a_structural_range() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::Float, &[5]);
        let key = f.key();
        let built = e_reduce_scalar(&mut f.ctx(), &a, ReduceKind::ArgMax, &key, "i").unwrap();
        let scalar = f.scalar_of(&built);
        let var = f
            .analysis
            .terms
            .var_of(scalar.term.as_int().unwrap())
            .expect("a fresh variable");
        let info = f.analysis.phi.var_info(var);
        assert_eq!(
            f.analysis.terms.constant_value(info.hi.unwrap()),
            Some(4),
            "0 <= v <= size(a) - 1"
        );
    }

    /// §5.7: a reduction over an axis drops it, or keeps it as 1.
    #[test]
    fn e_reduce_axis_drops_the_axis_or_keeps_it_as_one() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::Float, &[2, 3]);
        let dropped = e_reduce_axis(&mut f.ctx(), &a, ReduceKind::Sum, &[1], false).unwrap();
        assert_eq!(f.dims_of(&dropped), vec![Some(2)]);
        let kept = e_reduce_axis(&mut f.ctx(), &a, ReduceKind::Sum, &[1], true).unwrap();
        assert_eq!(f.dims_of(&kept), vec![Some(2), Some(1)]);
    }

    /// §5.7: `a @ b` contracts `K` against `K'` and keeps `[M, N]`.
    #[test]
    fn e_matmul_contracts_the_inner_dimensions() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::Float, &[2, 3]);
        let b = f.input(Dtype::Float, &[3, 4]);
        let built = e_matmul(&mut f.ctx(), &a, &b).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(2), Some(4)]);
        assert!(f.checks().is_empty(), "3 = 3 folds away");
    }

    /// §5.7's promotion rule B.5: a rank-1 operand becomes `[1, K]` or
    /// `[K, 1]` and the promoted axis is removed again.
    #[test]
    fn e_matmul_promotes_a_vector_operand() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::Float, &[3]);
        let b = f.input(Dtype::Float, &[3, 4]);
        let built = e_matmul(&mut f.ctx(), &a, &b).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(4)]);
    }

    /// §5.7's premise: rank ≥ 1.
    #[test]
    fn e_matmul_rejects_a_rank_zero_operand() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::Float, &[]);
        let b = f.input(Dtype::Float, &[3]);
        let outcome = e_matmul(&mut f.ctx(), &a, &b);
        assert!(
            matches!(
                outcome,
                Err(Reject::Premise {
                    rule: Rule::EMatmul,
                    ..
                })
            ),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.7: `sort`, `cumsum` and the reverse leave the dims alone;
    /// `argsort`'s elements are `Idx`.
    #[test]
    fn the_order_preserving_rules_leave_the_dims_alone() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::Float, &[4, 2]);
        for built in [
            e_sort(&mut f.ctx(), &a, false).unwrap(),
            e_cumsum(&mut f.ctx(), &a).unwrap(),
            e_reverse(&mut f.ctx(), &a).unwrap(),
        ] {
            assert_eq!(f.dims_of(&built), vec![Some(4), Some(2)]);
        }
        let indices = e_sort(&mut f.ctx(), &a, true).unwrap();
        assert_eq!(f.array_of(&indices).dtype, Dtype::Idx);
    }

    /// §5.7: `unique(a)` introduces its own count `c'` with `0 ≤ c' ≤
    /// size(a)`, plus the declared fact `size(a) > 0 ⇒ c' ≥ 1`.
    #[test]
    fn e_unique_introduces_an_independent_count() {
        let mut f = Fixture::new();
        let a = f.symbolic_input(Dtype::I64, "n", 16);
        let key = f.key();
        let built = e_unique(&mut f.ctx(), &a, &key, "c").unwrap();
        assert_eq!(f.dims_of(&built), vec![None]);
        let declared = f
            .analysis
            .phi
            .constraints()
            .iter()
            .filter(|c| matches!(c.kind, crate::phase0::phi::ConstraintKind::Declaration))
            .count();
        assert_eq!(declared, 1, "the non-empty fact of §5.7");
    }
}
