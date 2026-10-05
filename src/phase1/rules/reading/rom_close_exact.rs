//! `ROM_CLOSE_EXACT` (`revision/phase1/rules/reading/rom_close_exact.md`).
//!
//! Closes a ROM context: it takes the table of `ROM_BIND` and the ledger of
//! every selected `READ_ROM` against it, pads the record list to a power of
//! two, sorts it with a fixed bitonic network by `(pad, key, kind)`, and
//! checks that every real query record agrees in key and value with the
//! record before it.
//!
//! **`Q_o` is the ranged parameter.** The total query count of a context is
//! "the one integer phase 2 sets for a context" (`rules/README.md` §4.4):
//! `Q_o = Σ_r x_r·Q_r` over the selection variables of its candidate
//! readers. Phase 1 cannot know it — it is the sum over a selection phase 2
//! makes — so the node carries it as the one ranged parameter
//! `../../../README.md` §5.7 admits ([`crate::phase1::graph::Parameter`]),
//! with the range `[0, Σ_r Q_r]` over every reader the graph offers, and the
//! cost of the step function at its upper end, which is the largest the
//! checker can become.

use crate::phase1::build::{bits_for, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::interface::{Capacity, Family, Interface, OwnerRef};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct RomCloseExact;

impl Rule for RomCloseExact {
    fn id(&self) -> RuleId {
        RuleId::RomCloseExact
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::RomContext]
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
        let tables = builder.child(
            request,
            request.target.clone(),
            Family::RomTable,
            request.condition,
        )?;
        let mut out = Vec::new();
        for entry in &tables.entries.clone() {
            let Interface::RomTable { owner } = &entry.interface else {
                continue;
            };
            let owner = owner.clone();
            let mut candidate = Candidate::new(RuleId::RomCloseExact, Some("exact"), request.epoch);
            candidate.matched.push(value.clone());
            candidate.inputs.push(entry.requirement);
            candidate.epoch = candidate.epoch.max(entry.epoch);

            // The readers of this context are not known until the graph is
            // built, so the query count is the range phase 2 chooses within.
            let readers = self.readers(builder, &owner);
            let queries = readers.iter().map(|&(_, queries)| queries).sum::<Capacity>().max(1);
            candidate.readers = readers;
            let (comparators, records) = network(owner.slots(), queries);
            let width = super::key_width(&owner) + 2;
            candidate = candidate
                .emit(Primitive::Cmp { width }, comparators)
                .emit(Primitive::Select, 4 * comparators)
                .emit(Primitive::Mul, 3 * (records - 1).max(0) + 1)
                .ranged("Q_o", 0, queries)
                .parameter("B", owner.slots())
                .parameter("M", records)
                .parameter("w_sk", width as i128)
                .output(
                    request.target.clone(),
                    Interface::RomContext { owner: owner.clone() },
                    request.condition,
                );
            out.push(candidate);
        }
        Ok(out)
    }
}

impl RomCloseExact {
    /// The candidate readers of a context, with the queries each registers:
    /// the list the node carries and whose sum is the upper end of the range
    /// phase 2 chooses `Q_o` within (`rom_close_exact.md`,
    /// `resource_relations`).
    ///
    /// A reader of a context is a `READ_ROM` node against the same owner, and
    /// the graph holds one read request per read that reached this owner. The
    /// context node is built while those rows are being filled, so the list is
    /// the requests already in G plus the one being filled, which is the
    /// request that raised this context.
    fn readers(
        &self,
        builder: &Builder<'_>,
        owner: &OwnerRef,
    ) -> Vec<(crate::phase1::graph::ReqId, Capacity)> {
        builder
            .graph
            .requirements
            .iter()
            .enumerate()
            .filter_map(|(at, node)| match &node.id.form {
                crate::phase1::graph::Form::Represent { interface, .. } => match interface {
                    Interface::ReadReq { owner: named, queries, .. } if named == owner => {
                        Some((crate::phase1::graph::ReqId(at as u32), *queries))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }
}

/// `M = 2^{⌈log2(B_tot + Q_o)⌉}` and the comparators of the bitonic network
/// over it, `n_cmp(M) = M·m·(m + 1)/4` (`rom_close_exact.md`).
fn network(slots: Capacity, queries: Capacity) -> (Capacity, Capacity) {
    let total = slots + queries;
    if total <= 0 {
        return (0, 0);
    }
    let bits = bits_for(total) as Capacity;
    let records = 1i128 << bits.min(40);
    (records * bits * (bits + 1) / 4, records)
}
