//! `READ_KEY_MUX` (`revision/phase1/rules/reading/read_key_mux.md`).
//!
//! Answers a `Fetch` whose addresses are wires, or whose owner has run-time
//! keys, by selection over the whole owner: per entry one hit bit per slot,
//! the constraint "exactly one hit when active, none when inactive", and the
//! result as the sum of hit·value. Over a dense owner the hit bits are a
//! `onehot` of the address (variant `D`); over a ranked or keyed owner they
//! are equalities with the slots' keys (variant `C`). The cost is linear in
//! `Q · B_tot` with no shared state.

use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ReadKeyMux;

impl Rule for ReadKeyMux {
    fn id(&self) -> RuleId {
        RuleId::ReadKeyMux
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
            // Not instantiated where `READ_DIRECT` applies: its cost 0
            // dominates, and the two would be compared on equal children.
            if *constant && owner.is_dense() {
                continue;
            }
            let mut candidate = Candidate::new(
                RuleId::ReadKeyMux,
                Some(if owner.is_dense() { "D" } else { "C" }),
                request.epoch,
            );
            candidate.matched.push(value.clone());
            let Some(store) = super::store_port(builder, &mut candidate, request, owner)? else {
                continue;
            };
            candidate.inputs.push(entry.requirement);
            candidate.epoch = candidate.epoch.max(entry.epoch);
            candidate.inputs.push(store.0);
            candidate.epoch = candidate.epoch.max(store.1);

            let slots = owner.slots();
            candidate = if owner.is_dense() {
                // `D`: `onehot(addr_j over B_tot, act_j)` and the sum of
                // `e_{j,i} · values_i`.
                candidate
                    .emit(Primitive::OneHot { capacity: slots }, *queries)
                    .emit(Primitive::Mul, queries * slots)
            } else {
                // `C`: `eq` with each slot's key, gated by its validity and
                // by the entry's active flag.
                let gated = i128::from(!super::always_live(&batch));
                candidate
                    .emit(Primitive::Eq, queries * slots)
                    .emit(Primitive::Mul, queries * slots * (1 + gated))
            };
            candidate = candidate
                .answers_fetch(
                    request.target.clone(),
                    batch.clone(),
                    interface.clone(),
                    request.condition,
                )
                .parameter("Q", *queries)
                .parameter("B", slots);
            out.push(candidate);
        }
        Ok(out)
    }
}
