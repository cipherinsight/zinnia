//! `REINDEX_REF` (`revision/phase1/rules/mapping/reindex_ref.md`).
//!
//! Composes the two maps of a nested `Ref` into one, so that every reader
//! sees a single flat owner. Composition is symbolic: the outer map's
//! expressions are substituted into the inner map's and folded in the
//! grammar of `rules/README.md` §4.2, which admits every sum and literal
//! multiple of parameter wires, so no wire is allocated and the cost is 0.
//!
//! The four rows of the file's table are the four combinations of the two
//! forms; row 4, a coordinate map under a reshape, writes `unlin` out and
//! needs the inner extents beyond axis 0 to be literal, which is what
//! `../README.md` §4.2 admits as `div n` and `mod n`.
//!
//! **Where the nested bundle comes from.** The file gives the rule one port,
//! `Represent(v, Ref[v; o₁, μ₁])`, the nested member of v's own `Ref` row. A
//! rule that reads the row it is filling is exactly the cut of
//! `../../README.md` §4, "Acyclicity by construction", which would drop this
//! candidate and leave the row provisional. The rule therefore matches the
//! same source pattern as `MAKE_REF` and takes the *owner's* `Ref` row as
//! its port — the row μ₂ comes from — and composes the outer map it builds
//! itself with it. The node is the same: one `Ref` over the flat owner, the
//! composite map, cost 0.

use crate::phase1::build::Builder;
use crate::phase1::interface::{Capacity, Family, Mu, MuForm, RefIface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ReindexRef;

impl Rule for ReindexRef {
    fn id(&self) -> RuleId {
        RuleId::ReindexRef
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Ref]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(matched) = super::matches(builder, request) else {
            return Ok(Vec::new());
        };
        let mut probe = Candidate::new(RuleId::ReindexRef, None, request.epoch);
        let under = builder.under(request.condition);
        let Some(parameters) =
            super::parameters(builder, &mut probe, &matched, request.epoch, under)
        else {
            return Ok(Vec::new());
        };
        // The composite is constant exactly when π is empty; a run-time
        // parameter of the outer view is `MAKE_REF_RUNTIME`'s and not
        // composed here.
        let Some(starts) = parameters
            .iter()
            .map(|start| start.literal().map(super::MapParam::Literal))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(Vec::new());
        };
        super::make_ref::build_refs(
            builder,
            request,
            &matched,
            &mut probe,
            &starts,
            &[],
            RuleId::ReindexRef,
            /* flatten = */ true,
        )
    }
}

/// The composite map of a nested `Ref` (`reindex_ref.md`,
/// `produced_output_bundle`), by the four rows of its table.
pub(super) fn compose_maps(outer: &RefIface, inner: &RefIface) -> Option<(MuForm, Vec<Mu>)> {
    // `ext₂[b]`, the owner value's extents, literal where the table needs
    // them.
    let owner_extents: Option<Vec<Capacity>> = inner
        .ext
        .iter()
        .map(|extent| match extent {
            crate::phase1::interface::Extent::Literal(value) => Some(*value),
            crate::phase1::interface::Extent::Term(_) => None,
        })
        .collect();
    match (&outer.form, &inner.form) {
        // Row 1: coordinate under coordinate. Substitute the outer map's
        // expressions for the inner map's coordinate symbols.
        (MuForm::Coordinate { .. }, MuForm::Coordinate { strides }) => Some((
            MuForm::Coordinate { strides: strides.clone() },
            inner
                .map
                .iter()
                .map(|expression| substitute(expression, &outer.map))
                .collect(),
        )),
        // Row 2: coordinate under key. The composite is the linear position
        // of the outer map's coordinates in the owner value's C-order.
        (MuForm::Coordinate { .. }, MuForm::Key) => {
            let extents = owner_extents?;
            let mut strides = vec![1 as Capacity; extents.len()];
            for axis in (0..extents.len().saturating_sub(1)).rev() {
                strides[axis] = strides[axis + 1] * extents[axis + 1];
            }
            let mut sum: Option<Mu> = None;
            for (axis, expression) in outer.map.iter().enumerate() {
                let term = match strides.get(axis).copied().unwrap_or(1) {
                    1 => expression.clone(),
                    stride => Mu::Scale(stride, Box::new(expression.clone())),
                };
                sum = Some(match sum {
                    None => term,
                    Some(total) => Mu::Add(Box::new(total), Box::new(term)),
                });
            }
            Some((MuForm::Key, vec![sum.unwrap_or(Mu::Const(0))]))
        }
        // Row 3: key under key. C order preserves the linear position twice.
        (MuForm::Key, MuForm::Key) => Some((MuForm::Key, outer.map.clone())),
        // Row 4: key under coordinate. `unlin` is written out, by literal
        // divisors.
        (MuForm::Key, MuForm::Coordinate { strides }) => {
            let extents = owner_extents?;
            let position = outer.map.first()?.clone();
            let mut unlin = Vec::with_capacity(extents.len());
            for axis in 0..extents.len() {
                let stride: Capacity = extents[axis + 1..].iter().product();
                let mut expression = position.clone();
                if axis < extents.len() - 1 {
                    expression = Mu::DivConst(Box::new(expression), stride);
                }
                if axis > 0 {
                    expression = Mu::ModConst(Box::new(expression), extents[axis]);
                }
                unlin.push(expression);
            }
            Some((
                MuForm::Coordinate { strides: strides.clone() },
                inner
                    .map
                    .iter()
                    .map(|expression| substitute(expression, &unlin))
                    .collect(),
            ))
        }
    }
}

/// `e[j := μ₁(i)]`: the outer map's expressions substituted for the inner
/// map's coordinate symbols, folded in the grammar.
fn substitute(expression: &Mu, outer: &[Mu]) -> Mu {
    let folded = match expression {
        Mu::Coord(axis) => outer.get(*axis).cloned().unwrap_or(Mu::Const(0)),
        Mu::Add(left, right) => Mu::Add(
            Box::new(substitute(left, outer)),
            Box::new(substitute(right, outer)),
        ),
        Mu::Sub(left, right) => Mu::Sub(
            Box::new(substitute(left, outer)),
            Box::new(substitute(right, outer)),
        ),
        Mu::Scale(factor, inner) => Mu::Scale(*factor, Box::new(substitute(inner, outer))),
        Mu::ScaleParam(wire, inner) => {
            Mu::ScaleParam(*wire, Box::new(substitute(inner, outer)))
        }
        Mu::DivConst(inner, divisor) => {
            Mu::DivConst(Box::new(substitute(inner, outer)), *divisor)
        }
        Mu::ModConst(inner, modulus) => {
            Mu::ModConst(Box::new(substitute(inner, outer)), *modulus)
        }
        Mu::DivParam(inner, wire) => Mu::DivParam(Box::new(substitute(inner, outer)), *wire),
        Mu::ModParam(inner, wire) => Mu::ModParam(Box::new(substitute(inner, outer)), *wire),
        other => other.clone(),
    };
    fold(&folded)
}

/// Constant folding inside one μ expression, the "folded to Σ_k n_k × i_k + β"
/// of `reindex_ref.md`.
fn fold(expression: &Mu) -> Mu {
    match expression {
        Mu::Add(left, right) => match (fold(left), fold(right)) {
            (Mu::Const(a), Mu::Const(b)) => Mu::Const(a + b),
            (Mu::Const(0), other) | (other, Mu::Const(0)) => other,
            (a, b) => Mu::Add(Box::new(a), Box::new(b)),
        },
        Mu::Sub(left, right) => match (fold(left), fold(right)) {
            (Mu::Const(a), Mu::Const(b)) => Mu::Const(a - b),
            (a, Mu::Const(0)) => a,
            (a, b) => Mu::Sub(Box::new(a), Box::new(b)),
        },
        Mu::Scale(factor, inner) => match fold(inner) {
            Mu::Const(value) => Mu::Const(factor * value),
            other => Mu::Scale(*factor, Box::new(other)),
        },
        Mu::DivConst(inner, divisor) => match fold(inner) {
            Mu::Const(value) if *divisor != 0 => Mu::Const(value.div_euclid(*divisor)),
            other => Mu::DivConst(Box::new(other), *divisor),
        },
        Mu::ModConst(inner, modulus) => match fold(inner) {
            Mu::Const(value) if *modulus != 0 => Mu::Const(value.rem_euclid(*modulus)),
            other => Mu::ModConst(Box::new(other), *modulus),
        },
        other => other.clone(),
    }
}
