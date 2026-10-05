//! Merges at the end of a branch: `definition.md` §6.1.
//!
//! The merge $\sqcup_b$ is taken name by name. These functions merge one
//! pair of descriptors; walking the two scopes is
//! `crate::phase0::walk::env`'s job. A name defined on one side only is
//! undefined after the merge, which is a static error the walker reports.
//!
//! When `b` has origin shape, as in `if len(a) > 0`, the `ite` condition is a
//! shape term and phase 1 may prove it always true or always false, so the
//! branch's dimension terms collapse to one side. When `b` has origin data,
//! both sides are kept and phase 1's solver does the case split.

use super::Rule;
use super::{RuleCtx, VarRange};
use crate::phase0::desc::{
    ArrayDesc, CompositeDesc, Desc, Dtype, Origin, ScalarDesc, ScalarTerm, ValueKey,
};
use crate::phase0::term::BoolId;
use crate::phase0::Reject;

/// Merge two descriptors of the same name under the branch condition `b`.
pub fn merge_desc(
    ctx: &mut RuleCtx,
    b: BoolId,
    d1: &Desc,
    d2: &Desc,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    match (d1, d2) {
        (Desc::Array(a1), Desc::Array(a2)) => merge_array(ctx, b, a1, a2),
        (Desc::Scalar(s1), Desc::Scalar(s2)) => merge_scalar(ctx, b, s1, s2, value, base),
        (Desc::Composite(c1), Desc::Composite(c2)) => merge_composite(ctx, b, c1, c2, value, base),
        (Desc::None, Desc::None) => Ok(Desc::None),
        (Desc::Class(t1), Desc::Class(t2)) if t1 == t2 => Ok(Desc::Class(t1.clone())),
        _ => Err(Reject::MergeMismatch {
            detail: format!(
                "a {} on one side and a {} on the other",
                d1.kind_str(),
                d2.kind_str()
            ),
        }),
    }
}

/// Array merge (§6.1).
///
/// Premise: equal rank and equal dtype family; otherwise the program is
/// rejected (§1's rejection list). `dims[k] = ite(b, dims1[k], dims2[k])`,
/// `defcond = true` — both sides' conditions are already in $\Phi$ under
/// their path conditions — and $\psi$ is the $\psi$ at the merge point.
pub fn merge_array(
    ctx: &mut RuleCtx,
    b: BoolId,
    a1: &ArrayDesc,
    a2: &ArrayDesc,
) -> Result<Desc, Reject> {
    if a1.rank() != a2.rank() {
        return Err(Reject::MergeMismatch {
            detail: format!("rank {} against rank {}", a1.rank(), a2.rank()),
        });
    }
    let dtype = merged_dtype(a1.dtype, a2.dtype)?;
    let dims = a1
        .dims
        .iter()
        .zip(a2.dims.iter())
        .map(|(&p, &q)| ctx.terms.ite(b, p, q))
        .collect();
    let tt = ctx.tt();
    Ok(ctx.array(Rule::MergeArray, dtype, dims, tt))
}

/// Scalar merge (§6.1): the same term on both sides is kept; two shape
/// terms give `t = ite(b, t1, t2)` with origin `shape`; otherwise the result
/// is a fresh data variable `v`, and when both sides carry a term of one
/// sort the case split is kept as the declaration `φ ⇒ v = ite(b, t1, t2)`,
/// so that the solver sees which side `v` took instead of an unconstrained
/// variable. A side that carries no term (a `Float`) leaves `v` unconstrained.
///
/// Two sides that carry the *same* term merge to that term whatever their
/// origin: `ite(b, t, t) = t`, the identity fold the term constructors apply
/// to a dimension, applied here to the scalar. Without it a data scalar the
/// branch never touched would become a fresh variable with only its dtype
/// range, and the range its input declared would be lost. `Opaque` is the
/// absence of a term rather than a term the two sides share (§3.2: a `Float`,
/// `Complex` or `String` scalar carries none), so it does not fold.
pub fn merge_scalar(
    ctx: &mut RuleCtx,
    b: BoolId,
    s1: &ScalarDesc,
    s2: &ScalarDesc,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    let dtype = merged_dtype(s1.dtype, s2.dtype)?;
    let tt = ctx.tt();
    if s1.term == s2.term && s1.term != ScalarTerm::Opaque {
        let origin = Origin::join(s1.origin, s2.origin);
        return Ok(ctx.scalar(Rule::MergeScalar, dtype, s1.term, origin, tt));
    }
    if s1.origin == Origin::Shape && s2.origin == Origin::Shape {
        let merged = match (s1.term, s2.term) {
            (ScalarTerm::Int(t1), ScalarTerm::Int(t2)) => {
                Some(ScalarTerm::Int(ctx.terms.ite(b, t1, t2)))
            }
            (ScalarTerm::Bool(b1), ScalarTerm::Bool(b2)) => {
                Some(ScalarTerm::Bool(ctx.terms.bool_ite(b, b1, b2)))
            }
            _ => None,
        };
        if let Some(term) = merged {
            return Ok(ctx.scalar(Rule::MergeScalar, dtype, term, Origin::Shape, tt));
        }
    }
    let range = VarRange::of_dtype(ctx.terms, dtype);
    let merged = ctx.data_scalar(Rule::MergeScalar, dtype, base, value, range, tt);
    if let Some(selected) = selection(ctx, b, s1.term, s2.term, &merged) {
        ctx.declare(Rule::MergeScalar.detail("the selected side"), selected);
    }
    Ok(merged)
}

/// `v = ite(b, t1, t2)` for the fresh variable `v` of a data merge (§6.1),
/// or `None` when a side carries no term or the two terms differ in sort.
/// The Boolean sort has no `=` in the term language, so its equality is
/// spelled `(v ∧ w) ∨ (¬v ∧ ¬w)`.
fn selection(
    ctx: &mut RuleCtx,
    b: BoolId,
    t1: ScalarTerm,
    t2: ScalarTerm,
    merged: &Desc,
) -> Option<BoolId> {
    let v = merged.as_scalar()?.term;
    match (v, t1, t2) {
        (ScalarTerm::Int(v), ScalarTerm::Int(t1), ScalarTerm::Int(t2)) => {
            let selected = ctx.terms.ite(b, t1, t2);
            Some(ctx.terms.eq(v, selected))
        }
        (ScalarTerm::Bool(v), ScalarTerm::Bool(b1), ScalarTerm::Bool(b2)) => {
            let selected = ctx.terms.bool_ite(b, b1, b2);
            let both = ctx.terms.and(v, selected);
            let not_v = ctx.terms.not(v);
            let not_selected = ctx.terms.not(selected);
            let neither = ctx.terms.and(not_v, not_selected);
            Some(ctx.terms.or(both, neither))
        }
        _ => None,
    }
}

/// Composite merge (§6.1 applied to §3.3's composites): same kind and same
/// static length, merged element by element.
///
/// $\Delta$ keys the composite and not its elements (§3.3), so a fresh
/// variable this merge introduces would name the composite and nothing more.
/// The element index is pushed on [`RuleCtx::element`] around each element,
/// so every such variable records the path from the composite's key down to
/// the element that introduced it (§3.4), which is what lets a consumer name
/// the element.
///
/// Two lengths that differ would make the merged length depend on `b`, which
/// §3.3 forbids: a composite's length is a compile-time constant. That is the
/// `+` and `*` half of the rule [`crate::phase0::walk`]'s
/// `require_static_length` states for the list methods — `v = v + [x]` under
/// a data-dependent condition arrives here rather than there.
pub fn merge_composite(
    ctx: &mut RuleCtx,
    b: BoolId,
    c1: &CompositeDesc,
    c2: &CompositeDesc,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    if c1.kind != c2.kind {
        return Err(Reject::MergeMismatch {
            detail: format!("a {} against a {}", c1.kind.as_str(), c2.kind.as_str()),
        });
    }
    if c1.elems.len() != c2.elems.len() {
        return Err(Reject::MergeMismatch {
            detail: format!(
                "length {} against length {} under the condition `{}`: §3.3 makes a \
                 composite's length a compile-time constant, so it may not depend on a \
                 runtime value — build an array instead (`np.zeros(n)`), whose length \
                 may be value-sized (§5.9)",
                c1.elems.len(),
                c2.elems.len(),
                ctx.terms
                    .pretty_bool(b)
                    .chars()
                    .take(160)
                    .collect::<String>()
            ),
        });
    }
    let mut elems = Vec::with_capacity(c1.elems.len());
    for (i, (e1, e2)) in c1.elems.iter().zip(c2.elems.iter()).enumerate() {
        ctx.element.push(i);
        let merged = merge_desc(ctx, b, e1, e2, value, &format!("{}_{}", base, i));
        ctx.element.pop();
        elems.push(merged?);
    }
    Ok(Desc::Composite(CompositeDesc {
        kind: c1.kind,
        elems,
    }))
}

/// The dtype of a merge. §6.1 writes the premise as `τ1 = τ2`; §1's
/// rejection list states it as a dtype-*family* mismatch, which is the
/// weaker requirement this follows, so `Idx` and `I64` merge to `I64`.
fn merged_dtype(a: Dtype, b: Dtype) -> Result<Dtype, Reject> {
    if a.family() != b.family() {
        return Err(Reject::MergeMismatch {
            detail: format!("dtype {} against {}", a.as_str(), b.as_str()),
        });
    }
    Dtype::promote(a, b).ok_or_else(|| Reject::MergeMismatch {
        detail: format!("dtype {} against {}", a.as_str(), b.as_str()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::desc::CompositeKind;
    use crate::phase0::tests::Fixture;

    /// §6.1's array merge: `dims[k] = ite(b, dims1[k], dims2[k])`, and
    /// `defcond` is true.
    #[test]
    fn merge_array_joins_the_extents_under_the_condition() {
        let mut f = Fixture::new();
        let (a1, a2) = (f.input(Dtype::I64, &[2]), f.input(Dtype::I64, &[3]));
        let b = f.data_scalar(Dtype::Bool);
        let condition = match b.term {
            ScalarTerm::Bool(v) => v,
            _ => unreachable!("a Bool data scalar carries a Boolean term"),
        };
        let merged = merge_array(&mut f.ctx(), condition, &a1, &a2).unwrap();
        assert_eq!(f.dims_of(&merged), vec![None], "the extent is an ite");
        assert!(f.checks().is_empty(), "both sides' conditions are in Φ");
    }

    /// §6.1's premise: equal rank.
    #[test]
    fn merge_array_rejects_two_ranks() {
        let mut f = Fixture::new();
        let (a1, a2) = (f.input(Dtype::I64, &[2]), f.input(Dtype::I64, &[2, 2]));
        let tt = f.analysis.terms.tt();
        let outcome = merge_array(&mut f.ctx(), tt, &a1, &a2);
        assert!(
            matches!(outcome, Err(Reject::MergeMismatch { .. })),
            "got {:?}",
            outcome.err()
        );
    }

    /// §6.1's scalar merge: two shape terms give `ite(b, t1, t2)` with
    /// origin shape; a data side makes the result a fresh variable.
    #[test]
    fn merge_scalar_keeps_a_shape_term_and_falls_back_to_a_variable() {
        let mut f = Fixture::new();
        let (s1, s2) = (f.literal(2), f.literal(3));
        let b = f.data_scalar(Dtype::Bool);
        let condition = match b.term {
            ScalarTerm::Bool(v) => v,
            _ => unreachable!("a Bool data scalar carries a Boolean term"),
        };
        let key = f.key();
        let merged = merge_scalar(&mut f.ctx(), condition, &s1, &s2, &key, "m").unwrap();
        assert_eq!(f.scalar_of(&merged).origin, Origin::Shape);
        let data = f.data_scalar(Dtype::I64);
        let key = f.key();
        let mixed = merge_scalar(&mut f.ctx(), condition, &s1, &data, &key, "m").unwrap();
        assert_eq!(f.scalar_of(&mixed).origin, Origin::Data);
        let declared = f
            .analysis
            .phi
            .constraints()
            .iter()
            .filter(|c| matches!(c.kind, crate::phase0::phi::ConstraintKind::Declaration))
            .count();
        assert_eq!(declared, 1, "§6.1: the data merge declares v = ite(b, t1, t2)");
    }

    /// §3.3: a composite's length is a compile-time constant, so two
    /// lengths cannot merge.
    #[test]
    fn merge_composite_needs_the_same_kind_and_length() {
        let mut f = Fixture::new();
        let one = Desc::Scalar(f.literal(1));
        let short = CompositeDesc {
            kind: CompositeKind::List,
            elems: vec![one.clone()],
        };
        let long = CompositeDesc {
            kind: CompositeKind::List,
            elems: vec![one.clone(), one.clone()],
        };
        let tt = f.analysis.terms.tt();
        let key = f.key();
        let merged = merge_composite(&mut f.ctx(), tt, &short, &short, &key, "m").unwrap();
        assert_eq!(merged.as_composite().unwrap().elems.len(), 1);
        let key = f.key();
        let outcome = merge_composite(&mut f.ctx(), tt, &short, &long, &key, "m");
        assert!(
            matches!(outcome, Err(Reject::ValueSizedWithoutMaximum { .. }))
                || matches!(outcome, Err(Reject::MergeMismatch { .. })),
            "got {:?}",
            outcome.err()
        );
    }
}
