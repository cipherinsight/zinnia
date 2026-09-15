//! Composites, externals and properties: the pure forms of
//! `definition.md` §5.11.
//!
//! `E-Composite`, `E-Composite-Index`, `E-Composite-Binop`, `E-External` and
//! `E-Property`. The statement forms of §5.11 change the environment and so
//! live in the walker: `S-ListMethod` in `crate::phase0::walk::values`,
//! `S-SetItem` in `walk::subscript`, `S-Unpack` in `walk::stmts`.

use super::helpers::{constant_norm, size_term};
use super::inputs::{e_input, ContractSymbols};
use super::merge::merge_desc;
use super::Rule;
use super::{require_int_term, RuleCtx, VarRange};
use crate::phase0::desc::{
    ArrayDesc, CompositeDesc, CompositeKind, Desc, Dtype, Origin, ScalarDesc, ScalarTerm, ValueKey,
};
use crate::phase0::Reject;
use crate::types::ZinniaType;

/// `E-Composite` (§5.11): `[e_1, …, e_k]` or `(e_1, …, e_k)`, including
/// starred elements of known length, which the caller has already flattened.
pub fn e_composite(kind: CompositeKind, elems: Vec<Desc>) -> Desc {
    Desc::Composite(CompositeDesc { kind, elems })
}

/// `E-Composite-Index` (§5.11): `v[i]` with `v` a composite.
///
/// A constant `i` (negative wrapped) yields the element descriptor, and an
/// out-of-range constant rejects. A runtime `i` requires the elements to
/// agree up to terms; they are merged by `ite` over `i`, and
/// `defcond = (-k ≤ t_i < k)`.
pub fn e_composite_index(
    ctx: &mut RuleCtx,
    c: &CompositeDesc,
    index: &ScalarDesc,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    // The constant case returns an element descriptor it did not build, so
    // the rule tags itself rather than relying on a constructor.
    ctx.applied(Rule::ECompositeIndex);
    let k = c.elems.len() as i128;
    let t = require_int_term(Rule::ECompositeIndex, index)?;
    if let Some(n) = ctx.terms.constant_value(t) {
        let at = if n < 0 { n + k } else { n };
        if at < 0 || at >= k {
            return Err(Reject::Premise {
                rule: Rule::ECompositeIndex,
                detail: format!("with index {} into a composite of length {}", n, k),
            });
        }
        return Ok(c.elems[at as usize].clone());
    }
    if c.elems.is_empty() {
        return Err(Reject::Premise {
            rule: Rule::ECompositeIndex,
            detail: "with a runtime index into an empty composite".to_string(),
        });
    }
    // Fold from the last element backwards, guarding each with `at == j`.
    // The index is resolved first, so that `v[-1]` names the last element
    // rather than falling through to it by accident.
    let length = ctx.terms.constant(k);
    let resolved = super::helpers::wrap_index(ctx.terms, t, length);
    let mut merged = c.elems[c.elems.len() - 1].clone();
    for j in (0..c.elems.len() - 1).rev() {
        let at = ctx.terms.constant(j as i128);
        let hit = ctx.terms.eq(resolved, at);
        merged = merge_desc(
            ctx,
            hit,
            &c.elems[j],
            &merged,
            value,
            &format!("{}_{}", base, j),
        )?;
    }
    let defcond = super::helpers::in_bounds(ctx.terms, t, length);
    ctx.check(Rule::ECompositeIndex, defcond);
    Ok(merged)
}

/// A composite slice with constant bounds (§5.11): a sub-composite. Runtime
/// bounds reject.
pub fn e_composite_slice(
    ctx: &mut RuleCtx,
    c: &CompositeDesc,
    lo: Option<&ScalarDesc>,
    hi: Option<&ScalarDesc>,
    step: i64,
) -> Result<Desc, Reject> {
    ctx.applied(Rule::ECompositeIndex);
    let length = c.elems.len() as i64;
    let bound = |ctx: &mut RuleCtx, s: Option<&ScalarDesc>| -> Result<Option<i64>, Reject> {
        match s {
            None => Ok(None),
            Some(d) => {
                let t = require_int_term(Rule::ECompositeIndex, d)?;
                match ctx.terms.constant_value(t) {
                    Some(n) => Ok(Some(n as i64)),
                    None => Err(Reject::Premise {
                        rule: Rule::ECompositeIndex,
                        detail: "with a runtime bound".to_string(),
                    }),
                }
            }
        }
    };
    let (lo, hi) = (bound(ctx, lo)?, bound(ctx, hi)?);
    if step == 0 {
        return Err(Reject::Premise {
            rule: Rule::ECompositeIndex,
            detail: "with step 0".to_string(),
        });
    }
    let indices = python_slice_indices(lo, hi, step, length);
    let elems = indices.into_iter().map(|i| c.elems[i].clone()).collect();
    Ok(e_composite(c.kind, elems))
}

/// The positions Python's `v[lo:hi:step]` visits, using the same clamping as
/// [`super::helpers::norm`] but on constants.
pub(crate) fn python_slice_indices(
    lo: Option<i64>,
    hi: Option<i64>,
    step: i64,
    length: i64,
) -> Vec<usize> {
    let negative = step < 0;
    // The clamp is `E-Slice`'s `norm` on literals, so that a composite slice
    // and an array slice cannot disagree (§5.4, §5.11). It is `pub(crate)`
    // only so that the CPython table can drive it beside `norm` and
    // `range_len` (`tests::norm_and_slice_length_match_python_on_constants`).
    let d = length as i128;
    let clamp = |b: i64| constant_norm(b as i128, d, negative) as i64;
    let (lower, upper) = if negative {
        (-1, length - 1)
    } else {
        (0, length)
    };
    let start = lo
        .map(clamp)
        .unwrap_or(if negative { upper } else { lower });
    let stop = hi
        .map(clamp)
        .unwrap_or(if negative { lower } else { upper });
    let mut out = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        out.push(i as usize);
        i += step;
    }
    out
}

/// `E-Composite-Binop` (§5.11): `v + w` concatenates, `v * n` and `n * v`
/// with a constant `n` repeat, `v == w` collapses to one Boolean.
pub fn e_composite_binop(
    ctx: &mut RuleCtx,
    op: CompositeBinop,
    v: &CompositeDesc,
    value: &ValueKey,
    base: &str,
) -> Result<Desc, Reject> {
    ctx.applied(Rule::ECompositeBinop);
    match op {
        CompositeBinop::Concat(w) => {
            let mut elems = v.elems.clone();
            elems.extend(w.elems.iter().cloned());
            Ok(e_composite(v.kind, elems))
        }
        CompositeBinop::Repeat(n) => {
            let count = n.max(0) as usize;
            let mut elems = Vec::with_capacity(v.elems.len() * count);
            for _ in 0..count {
                elems.extend(v.elems.iter().cloned());
            }
            Ok(e_composite(v.kind, elems))
        }
        CompositeBinop::Equal(_) => {
            let tt = ctx.tt();
            let range = VarRange::default();
            Ok(ctx.data_scalar(Rule::ECompositeBinop, Dtype::Bool, base, value, range, tt))
        }
    }
}

/// The three composite binary operators of §5.11.
pub enum CompositeBinop<'d> {
    Concat(&'d CompositeDesc),
    Repeat(i64),
    Equal(&'d CompositeDesc),
}

/// `E-External` (§5.11): `f(args)` for an `@zk_external f`. The result
/// descriptor comes from the declared return type; arguments have no shape
/// effect.
pub fn e_external(
    ctx: &mut RuleCtx,
    name: &str,
    return_dt: &ZinniaType,
    value: &ValueKey,
    symbols: &mut ContractSymbols,
) -> Result<Desc, Reject> {
    // The descriptor is `E-Input-*`'s, which tags itself; this rule names the
    // reason it was built here.
    ctx.applied(Rule::EExternal);
    e_input(ctx, name, value, return_dt, symbols)
}

/// The array properties of §5.11's `E-Property`.
#[derive(Debug, Clone, Copy)]
pub enum Property {
    Shape,
    Ndim,
    Size,
    Dtype,
}

/// `E-Property` (§5.11): `a.shape`, `a.ndim`, `a.size`, `a.dtype`.
///
/// All of them are shape scalars read off the descriptor; `a.shape` is a
/// tuple composite of them. `len(a)` is `E-Len` (§5.2).
pub fn e_property(
    ctx: &mut RuleCtx,
    a: &ArrayDesc,
    property: Property,
    value: &ValueKey,
) -> Result<Desc, Reject> {
    let tt = ctx.tt();
    Ok(match property {
        Property::Shape => {
            let elems = a
                .dims
                .clone()
                .into_iter()
                .map(|d| {
                    ctx.scalar(
                        Rule::EProperty,
                        Dtype::Idx,
                        ScalarTerm::Int(d),
                        Origin::Shape,
                        tt,
                    )
                })
                .collect();
            e_composite(CompositeKind::Tuple, elems)
        }
        Property::Ndim => {
            let t = ctx.terms.constant(a.rank() as i128);
            ctx.scalar(
                Rule::EProperty,
                Dtype::Idx,
                ScalarTerm::Int(t),
                Origin::Shape,
                tt,
            )
        }
        Property::Size => {
            let t = size_term(ctx, &a.dims, value, "sigma_size");
            ctx.scalar(
                Rule::EProperty,
                Dtype::Idx,
                ScalarTerm::Int(t),
                Origin::Shape,
                tt,
            )
        }
        Property::Dtype => Desc::Class(a.dtype.to_zinnia_type()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::tests::Fixture;

    fn pair(f: &mut Fixture) -> CompositeDesc {
        let (a, b) = (f.literal(10), f.literal(20));
        CompositeDesc {
            kind: CompositeKind::Tuple,
            elems: vec![Desc::Scalar(a), Desc::Scalar(b)],
        }
    }

    /// §5.11: a constant index gives the element descriptor; a negative one
    /// wraps.
    #[test]
    fn e_composite_index_selects_by_a_constant_index() {
        let mut f = Fixture::new();
        let c = pair(&mut f);
        let one = f.literal(1);
        let key = f.key();
        let picked = e_composite_index(&mut f.ctx(), &c, &one, &key, "v").unwrap();
        let term = f.scalar_of(&picked).term.as_int().unwrap();
        assert_eq!(f.analysis.terms.constant_value(term), Some(20));
        let back = f.literal(-1);
        let key = f.key();
        let last = e_composite_index(&mut f.ctx(), &c, &back, &key, "v").unwrap();
        let term = f.scalar_of(&last).term.as_int().unwrap();
        assert_eq!(f.analysis.terms.constant_value(term), Some(20));
    }

    /// §5.11: "out of range rejects", as a premise of the rule.
    #[test]
    fn e_composite_index_out_of_range_is_a_premise_violation() {
        let mut f = Fixture::new();
        let c = pair(&mut f);
        let two = f.literal(2);
        let key = f.key();
        let outcome = e_composite_index(&mut f.ctx(), &c, &two, &key, "v");
        assert!(
            matches!(
                outcome,
                Err(Reject::Premise {
                    rule: Rule::ECompositeIndex,
                    ..
                })
            ),
            "got {:?}",
            outcome.err()
        );
    }

    /// §5.11: a slice with constant bounds is a sub-composite.
    #[test]
    fn e_composite_slice_takes_the_python_sub_list() {
        let mut f = Fixture::new();
        let c = pair(&mut f);
        let one = f.literal(1);
        let sliced = e_composite_slice(&mut f.ctx(), &c, Some(&one), None, 1).unwrap();
        let elems = sliced.as_composite().expect("a composite").elems.clone();
        assert_eq!(elems.len(), 1);
    }

    /// §5.11: `v + w` concatenates, `v * n` repeats, `v == w` is one
    /// Boolean.
    #[test]
    fn e_composite_binop_adds_and_multiplies_lengths() {
        let mut f = Fixture::new();
        let c = pair(&mut f);
        let other = pair(&mut f);
        let key = f.key();
        let joined =
            e_composite_binop(&mut f.ctx(), CompositeBinop::Concat(&other), &c, &key, "v").unwrap();
        assert_eq!(joined.as_composite().unwrap().elems.len(), 4);
        let key = f.key();
        let repeated =
            e_composite_binop(&mut f.ctx(), CompositeBinop::Repeat(3), &c, &key, "v").unwrap();
        assert_eq!(repeated.as_composite().unwrap().elems.len(), 6);
        let key = f.key();
        let compared =
            e_composite_binop(&mut f.ctx(), CompositeBinop::Equal(&other), &c, &key, "v").unwrap();
        assert_eq!(f.scalar_of(&compared).dtype, Dtype::Bool);
    }

    /// §5.11: `a.shape` is a tuple of shape scalars, `a.ndim` and `a.size`
    /// are shape scalars, `a.dtype` is the dtype object.
    #[test]
    fn e_property_reads_the_descriptor() {
        let mut f = Fixture::new();
        let a = f.input(Dtype::Float, &[2, 3]);
        let key = f.key();
        let shape = e_property(&mut f.ctx(), &a, Property::Shape, &key).unwrap();
        assert_eq!(shape.as_composite().unwrap().elems.len(), 2);
        let ndim = e_property(&mut f.ctx(), &a, Property::Ndim, &key).unwrap();
        let term = f.scalar_of(&ndim).term.as_int().unwrap();
        assert_eq!(f.analysis.terms.constant_value(term), Some(2));
        let size = e_property(&mut f.ctx(), &a, Property::Size, &key).unwrap();
        let term = f.scalar_of(&size).term.as_int().unwrap();
        assert_eq!(f.analysis.terms.constant_value(term), Some(6));
        let dtype = e_property(&mut f.ctx(), &a, Property::Dtype, &key).unwrap();
        assert!(matches!(dtype, Desc::Class(ZinniaType::Float)));
    }

    /// §5.11: `E-External` takes the declared return type's descriptor.
    #[test]
    fn e_external_takes_the_declared_return_type() {
        let mut f = Fixture::new();
        let mut symbols = ContractSymbols::default();
        let key = f.key();
        let dt = ZinniaType::NDArray {
            shape: vec![3, 2],
            dtype: crate::types::NumberType::Float,
        };
        let built = e_external(&mut f.ctx(), "oracle", &dt, &key, &mut symbols).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(3), Some(2)]);
    }
}
