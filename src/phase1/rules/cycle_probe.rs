//! A rule that exists only in the tests, to exercise the cut of
//! `revision/phase1/README.md` §4, "Acyclicity by construction".
//!
//! Among the scalar rules no cycle occurs: every rule reads values recorded
//! earlier in Δ, and Δ's order is a topological order of the source graph.
//! The array rules do close one — `BUILD_DENSE` over a `Fetch` of v that a
//! read of v's own bundle would answer, which
//! `tests::arrays::reading_a_value_to_rebuild_it_is_cut` exercises — but the
//! cut on a *scalar* row still has no program that reaches it, so this rule
//! closes one on purpose: it answers `Represent(v, Wire[τ])` for a scalar
//! value of Δ bound to the name `cycle` by requiring that same requirement.
//!
//! It matches nothing else, so every other test sees the library without
//! it.

use crate::phase1::build::Builder;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

/// The name a value must be bound to for this rule to match.
pub(crate) const MARKER: &str = "cycle";

pub(crate) struct CycleProbe;

impl Rule for CycleProbe {
    fn id(&self) -> RuleId {
        RuleId::CycleProbe
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Target::Value(key) = &request.target else {
            return Ok(Vec::new());
        };
        let Some(entry) = builder.entry(key) else {
            return Ok(Vec::new());
        };
        // The value the marked name is bound to. A binding that only
        // relabels its right-hand side has the rows of that side
        // (`rules/README.md` §2), so the row this rule has to match is the
        // side's and not the binding's.
        let marked = builder.result.descriptors.iter().any(|other| {
            other.name.as_deref() == Some(MARKER)
                && (&other.key == key || other.producer.relabels() == Some(key))
        });
        if !marked {
            return Ok(Vec::new());
        }
        let Some(scalar) = entry.desc.as_scalar() else {
            return Ok(Vec::new());
        };
        let dtype = scalar.dtype;
        let mut candidate = Candidate::new(RuleId::CycleProbe, None, request.epoch);
        candidate.matched.push(key.clone());
        // The port is the row being filled, so the table's in-progress stack
        // answers it with the cut: an empty row, which drops this candidate
        // and makes the row provisional.
        let row = builder.child(
            request,
            request.target.clone(),
            Family::Wire,
            request.condition,
        )?;
        let Some(first) = row.entries.first() else {
            return Ok(Vec::new());
        };
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
        candidate = candidate.output(
            request.target.clone(),
            Interface::Wire(dtype),
            request.condition,
        );
        Ok(vec![candidate])
    }
}
