//! `READ_DIRECT` (`revision/phase1/rules/reading/read_direct.md`).
//!
//! Answers a `Fetch` by wiring: when every address of the read request is a
//! compile-time constant and the owner is dense (box or compact, so that the
//! key of a slot is the slot), the slot holding each entry is known at
//! compile time and the results are the owner's value wires themselves. It
//! allocates nothing and emits no constraint. `../../README.md` §4 step 4
//! makes it the mechanism whenever every composed coordinate is CONST.

use crate::phase1::build::Builder;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ReadDirect;

impl Rule for ReadDirect {
    fn id(&self) -> RuleId {
        RuleId::ReadDirect
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Values]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let (Some(value), Some(batch)) = (request.target.as_value(), request.batch.clone()) else {
            return Ok(Vec::new());
        };
        let value = value.clone();
        let Some(interface) = super::values(builder, &value, &batch) else {
            return Ok(Vec::new());
        };
        let requests = builder.child_in_batch(
            request,
            request.target.clone(),
            Family::ReadReq,
            request.condition,
        )?;
        let mut out = Vec::new();
        for entry in &requests.entries.clone() {
            let Interface::ReadReq { owner, queries, constant } = &entry.interface else {
                continue;
            };
            // D1, from the record: every address is a constant. D2: the owner
            // is dense, so the slot of a constant address is a constant; a
            // ranked or keyed owner has run-time keys and no fixed slot.
            if !constant || !owner.is_dense() {
                continue;
            }
            let mut candidate = Candidate::new(RuleId::ReadDirect, None, request.epoch);
            candidate.matched.push(value.clone());
            let Some(store) = super::store_port(builder, &mut candidate, request, owner)? else {
                continue;
            };
            candidate.inputs.push(entry.requirement);
            candidate.epoch = candidate.epoch.max(entry.epoch);
            candidate.inputs.push(store.0);
            candidate.epoch = candidate.epoch.max(store.1);
            candidate = candidate
                .answers_fetch(
                    request.target.clone(),
                    batch.clone(),
                    interface.clone(),
                    request.condition,
                )
                .parameter("Q", *queries);
            out.push(candidate);
        }
        Ok(out)
    }
}
