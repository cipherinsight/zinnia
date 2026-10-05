//! `INDEX_NORMALIZE`
//! (`revision/phase1/rules/scalar_control/index_normalize.md`).
//!
//! Takes a scalar index i and an axis extent d and produces the normalised
//! position `j = ite(i < 0, i + d, i)` and the bit `ok = (0 ≤ j < d)`. It is
//! `TERM_TO_WIRES` on those two terms, kept as one rule so that a reader that
//! needs j as an address and a check that needs ok name one node (§3.4), and
//! so that ok costs two comparisons instead of three.
//!
//! The variants are decided on intervals, with no solver: `/const` (both
//! literal), `/nonneg` (`lo(i) ≥ 0`: j = i), `/neg` (`hi(i) < 0`: j = i + d),
//! `/full` otherwise.

use crate::phase0::desc::Dtype;
use crate::phase0::interval::Interval;
use crate::phase0::term::TermId;

use crate::phase1::build::{width_of, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct IndexNormalize;

/// The `(i, d)` of phase 0's in-bounds condition `−d ≤ i ∧ i < d`, when a
/// Boolean term is one (`../../phase0/rules/helpers.rs`, `in_bounds`).
pub(crate) fn in_bounds_pattern(
    builder: &Builder<'_>,
    formula: crate::phase0::term::BoolId,
) -> Option<(TermId, TermId)> {
    use crate::phase0::term::NodeView;
    let terms = &builder.terms;
    let NodeView::And(low, high) = terms.node_at(formula.index()) else {
        return None;
    };
    let NodeView::Le(negated, left) = terms.node_at(low as usize) else {
        return None;
    };
    let NodeView::Lt(right, extent) = terms.node_at(high as usize) else {
        return None;
    };
    if left != right {
        return None;
    }
    // `−d` is `mul_const(-1, d)`, which folds to a literal when d is one.
    let matches_extent = match (terms.node_at(negated as usize), terms.node_at(extent as usize)) {
        (NodeView::Mul(-1, inner), _) => inner == extent,
        (NodeView::Const(a), NodeView::Const(b)) => a == -b,
        _ => false,
    };
    if !matches_extent {
        return None;
    }
    Some((terms.int_at(left)?, terms.int_at(extent)?))
}

impl Rule for IndexNormalize {
    fn id(&self) -> RuleId {
        RuleId::IndexNormalize
    }

    fn answers(&self) -> &'static [Family] {
        // §3.4: the pair `(j, ok)` is one set of wires with two projections,
        // and the `ok` bit is the wire of the Boolean term `−d ≤ t ∧ t < d`
        // that `ENFORCE_CHECK` reads (`term_to_wires.md` P3).
        &[Family::IndexNorm, Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        // The requirement names the pair ⟨i, d⟩: the index is the row's
        // target and the extent the interface's parameter, which is what a
        // reader raising the port fills in. A row with no extent is no
        // instance of this rule.
        let Some((index, extent, bit)) = self.subject(builder, request) else {
            return Ok(Vec::new());
        };
        let mut candidate = Candidate::new(RuleId::IndexNormalize, None, request.epoch);
        if let Target::Value(key) = &request.target {
            candidate.matched.push(key.clone());
        }
        // `idx`: the index's own row. `ext`: the extent's wire, absent when
        // the extent is a constant.
        let row = builder.child(
            request,
            Target::Term(index),
            Family::Wire,
            request.condition,
        )?;
        let Some(first) = row.entries.first() else {
            return Ok(Vec::new());
        };
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
        let literal_extent = builder.terms.constant_value(extent);
        if literal_extent.is_none() {
            let row = builder.child(
                request,
                Target::Term(extent),
                Family::Wire,
                request.condition,
            )?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }

        // P1 (variant): the interval facts.
        let range = builder.interval(index);
        let extent_range = builder.interval(extent);
        let (variant, primitives) = self.rows(builder, range, extent_range, literal_extent);
        for (primitive, times) in primitives {
            candidate = candidate.emit(primitive, times);
        }
        candidate.variant = Some(variant);
        // §3.4: the normalised position and the in-range bit are two
        // projections of one set of wires. The position is the wire a reader
        // addresses with; the bit is what `ENFORCE_CHECK` reads.
        //
        // The pair is attached to the row's own target: `source_pattern`
        // admits the index as a value of Δ or as a term, so a consumer that
        // names the scalar by its key — `SETITEM`'s `idx_k`, the rank-0
        // index of `TERM_TO_WIRES`'s β_I — gets its node in the row it
        // asked for and not in the term's.
        candidate = candidate.output(
            request.target.clone(),
            Interface::IndexNorm { extent },
            request.condition,
        );
        if let Some(bit) = bit {
            candidate = candidate.output(
                Target::BoolTerm(bit),
                Interface::Wire(Dtype::Bool),
                request.condition,
            );
        }
        Ok(vec![candidate])
    }
}

impl IndexNormalize {
    /// What the row is about: the index term, the extent, and — when the row
    /// is the `ok` bit's — the Boolean term that bit holds.
    ///
    /// An `IndexNorm` row names the pair `⟨i, d⟩` directly. A `Wire` row over
    /// a Boolean term is this rule's when the term is phase 0's in-bounds
    /// condition `−d ≤ t ∧ t < d` (`../../phase0/rules/helpers.rs`,
    /// `in_bounds`), which is P3 of `term_to_wires.md`.
    fn subject(
        &self,
        builder: &Builder<'_>,
        request: &RowRequest,
    ) -> Option<(TermId, TermId, Option<crate::phase0::term::BoolId>)> {
        match &request.target {
            Target::BoolTerm(formula) => {
                let (index, extent) = in_bounds_pattern(builder, *formula)?;
                Some((index, extent, Some(*formula)))
            }
            Target::Value(key) => {
                let index = builder.scalar(key).and_then(|scalar| scalar.term.as_int())?;
                Some((index, request.extent?, None))
            }
            Target::Term(term) => Some((*term, request.extent?, None)),
        }
    }

    /// The emitter rows of the variant the intervals select
    /// (`index_normalize.md`, emitter).
    fn rows(
        &self,
        builder: &Builder<'_>,
        index: Interval,
        extent: Interval,
        literal_extent: Option<i128>,
    ) -> (&'static str, Vec<(Primitive, i128)>) {
        let difference = |a: Interval, b: Interval| super::difference(a, b);
        let sign_width = builder.omega.round(width_of(index));
        let high_width = builder.omega.round(width_of(difference(index, extent)));
        let low_width = builder.omega.round(width_of(crate::phase0::interval::Interval::new(
            index.lo.zip(extent.hi).map(|(lo, hi)| lo + hi),
            index.hi.zip(extent.lo).map(|(hi, lo)| hi + lo),
        )));
        // `/const`: both the index and the extent are pinned, so the position
        // and the bit are constants and nothing is emitted.
        if index.lo == index.hi && index.lo.is_some() && literal_extent.is_some() {
            return ("/const", Vec::new());
        }
        if index.lo.unwrap_or(-1) >= 0 {
            // `/nonneg`: j is the port's wire, ok one comparison.
            return ("/nonneg", vec![(Primitive::Cmp { width: high_width }, 1)]);
        }
        if index.hi.unwrap_or(0) < 0 {
            return ("/neg", vec![(Primitive::Cmp { width: low_width }, 1)]);
        }
        (
            "/full",
            vec![
                (Primitive::Cmp { width: sign_width }, 1),
                (Primitive::Mul, i128::from(literal_extent.is_none())),
                (Primitive::Cmp { width: low_width }, 1),
                (Primitive::Cmp { width: high_width }, 1),
                (Primitive::And, 1),
            ],
        )
    }
}


