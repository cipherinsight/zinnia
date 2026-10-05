//! `ROM_BIND` (`revision/phase1/rules/reading/rom_bind.md`).
//!
//! Turns one Store-family node of a value into a read-only table: one init
//! record per slot, `(pad, key, kind = 0, value)`, and opens the ledger the
//! `READ_ROM` nodes register their query records in. It fixes the identity of
//! the context, `(o, B)` — the owner node and its capacity — and the key
//! width every reader and the checker use. It allocates nothing beyond linear
//! forms, so its cost is 0.

use crate::phase1::build::Builder;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct RomBind;

impl Rule for RomBind {
    fn id(&self) -> RuleId {
        RuleId::RomBind
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::RomTable]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(value) = request.target.as_value() else {
            return Ok(Vec::new());
        };
        let value = value.clone();
        let mut out = Vec::new();
        // source_pattern: a value with a Store-family provider node. The
        // producer is not read; one node per provider node.
        for family in [Family::Store, Family::Masked, Family::Packed] {
            let row = builder.child(request, request.target.clone(), family, request.condition)?;
            for entry in &row.entries {
                let Some(owner) = super::owner_of(&entry.interface, &value) else {
                    continue;
                };
                // premises (structural): a ranked or keyed owner has rank 1,
                // so that its key is the logical index (`README.md` §3.2).
                if !owner.is_dense() && builder.array(&value).map(|a| a.rank()) != Some(1) {
                    continue;
                }
                let mut candidate = Candidate::new(RuleId::RomBind, None, request.epoch);
                candidate.matched.push(value.clone());
                candidate.inputs.push(entry.requirement);
                candidate.epoch = candidate.epoch.max(entry.epoch);
                candidate = candidate
                    .output(
                        request.target.clone(),
                        Interface::RomTable { owner: owner.clone() },
                        request.condition,
                    )
                    .parameter("B", owner.slots())
                    .parameter("w_key", super::key_width(&owner) as i128);
                out.push(candidate);
            }
        }
        Ok(out)
    }
}
