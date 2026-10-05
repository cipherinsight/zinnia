//! `READ_ROM` (`revision/phase1/rules/reading/read_rom.md`).
//!
//! Answers a `Fetch` by result witnesses: the prover supplies the Q results,
//! and the node registers Q query records in the ledger of the ROM context of
//! the owner, where `ROM_CLOSE_EXACT` checks every registered record at once
//! against the table. The context is a *port*, so ROM closure is structural
//! (`../../README.md` §2, "Edges"): selecting this node forces the context,
//! hence the table and the checker, into the subgraph.

use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ReadRom;

impl Rule for ReadRom {
    fn id(&self) -> RuleId {
        RuleId::ReadRom
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
            if *constant && owner.is_dense() {
                continue;
            }
            let mut candidate = Candidate::new(RuleId::ReadRom, None, request.epoch);
            candidate.matched.push(value.clone());
            // `ctx`: the context node of the owner, which `ROM_CLOSE_EXACT`
            // answers. It is required, not read: that is what closes the
            // ledger structurally.
            let psi = builder.psi(&owner.value);
            let condition =
                builder.child_under_g(&mut candidate, request, psi, &owner.value.encode())?;
            let contexts = builder.child(
                request,
                Target::Value(owner.value.clone()),
                Family::RomContext,
                condition,
            )?;
            let Some(context) = contexts
                .entries
                .iter()
                .find(|context| matches!(&context.interface,
                    Interface::RomContext { owner: named } if named == owner))
                .cloned()
            else {
                continue;
            };
            candidate.inputs.push(entry.requirement);
            candidate.epoch = candidate.epoch.max(entry.epoch);
            candidate.inputs.push(context.requirement);
            candidate.epoch = candidate.epoch.max(context.epoch);

            // emitter, per registered entry: the masking `select` when both
            // the address and the flag are wires, the range check on the
            // key, and the product that forces an inactive result to 0.
            let width = super::key_width(owner);
            let live = super::always_live(&batch);
            let gated = i128::from(!live);
            candidate = candidate
                .emit(Primitive::Select, queries * gated)
                .emit(Primitive::Range { width }, *queries)
                .emit(Primitive::Mul, queries * gated)
                .emit(Primitive::Witness, *queries)
                .answers_fetch(
                    request.target.clone(),
                    batch.clone(),
                    interface.clone(),
                    request.condition,
                )
                .parameter("Q", *queries)
                .parameter("w_key", width as i128);
            out.push(candidate);
        }
        Ok(out)
    }
}
