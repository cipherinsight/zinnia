//! `PREFIX_RANK` (`revision/phase1/rules/masks/prefix_rank.md`).
//!
//! Takes a value m held in an ordered bundle I of capacity B and computes,
//! per physical position i, the active bit `t_i` and the rank
//! `r_i = Σ_{j<i} t_j`. Variant `mask` ranks a Boolean value's elements,
//! `t_i = valid_I(i) ∧ m_i`, and its last rank `r_B` is the count variable
//! `c = Count(m)`, so the node is attached to two requirements (§3.4);
//! variant `valid` ranks the bundle's own validity wires, which is
//! `Ranks(I, valid)` of `rules/README.md` §4.5 — over a `Packed` the
//! thermometer of ℓ, built once here and shared by every consumer that holds
//! the bundle.
//!
//! The rule asks no question: every premise is structural.
//!
//! **One candidate, two rows.** The node answers the `Ranks(I)` row of m and
//! the `Wire[Idx]` row of c, which the construction fills separately. Both
//! rows build the candidate through [`node`], with the same ports,
//! parameters, epoch and cost, so `Graph::add_implementation` recognises the
//! two offers as one node ([`crate::phase1::rules::masks::count`]).

use crate::phase0::desc::{Dtype, ValueKey};

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Family, Interface, RanksOf};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct PrefixRank;

impl Rule for PrefixRank {
    fn id(&self) -> RuleId {
        RuleId::PrefixRank
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Ranks]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        // The row names the bundle it ranks: `Ranks(I)` and `Ranks(I, valid)`
        // over two bundles of one value are four rows (§4.5).
        let (Some(value), Some(subject)) = (request.target.as_value(), request.ranks.clone())
        else {
            return Ok(Vec::new());
        };
        let value = value.clone();
        Ok(node(builder, request, &value, &subject)?.into_iter().collect())
    }
}

/// The `PREFIX_RANK` node over one bundle of one value, or `None` when the
/// premises of `prefix_rank.md` fail.
///
/// Both rows the node answers call this, so that the candidate offered in
/// each is the same instance (§3.4).
pub(crate) fn node(
    builder: &mut Builder<'_>,
    request: &RowRequest,
    value: &ValueKey,
    subject: &RanksOf,
) -> Result<Option<Candidate>, Reject> {
    let Some(array) = builder.array(value).cloned() else {
        return Ok(None);
    };
    // premises (structural): rank ≥ 1; for `mask`, dtype Bool; the layout is
    // ordered, so the valid positions in physical order are the elements in
    // logical order.
    if array.rank() == 0 || !super::ordered(&subject.owner) {
        return Ok(None);
    }
    if !subject.valid && array.dtype != Dtype::Bool {
        return Ok(None);
    }
    let variant = if subject.valid { "valid" } else { "mask" };
    // The node represents m, so its default epoch is m's watermark
    // (`rules/README.md` §3.7) — not the row's, which differs between the
    // `Ranks` row and the count's wire row.
    let mut candidate = Candidate::new(RuleId::PrefixRank, Some(variant), builder.watermark(value));
    candidate.matched.push(value.clone());

    // `mask`: the bundle I, raised under ψ_I.
    let Some(port) =
        crate::phase1::rules::reading::store_port(builder, &mut candidate, request, &subject.owner)?
    else {
        return Ok(None);
    };
    candidate.inputs.push(port.0);
    candidate.epoch = candidate.epoch.max(port.1);
    let condition = builder.psi(value);

    // Over a `Packed` whose tail is not zeroed the active bit of a position
    // is `valid_I(i) ∧ m_i`, and its validity is the thermometer of ℓ, which
    // `rules/README.md` §4.1 makes the `Ranks(I, valid)` node of the same
    // bundle rather than a wire of this one: the `mask` variant requires it
    // and emits no thermometer. Over a canonical `Packed` the value wire is
    // already the active bit, so no validity is read.
    if !subject.valid && super::validity_is_a_thermometer(&subject.owner) && !subject.owner.canon {
        let validity = RanksOf { owner: subject.owner.clone(), valid: true };
        let row = builder.ranks_port(&mut candidate, request, value, validity)?;
        let Some(first) = row.entries.first() else {
            return Ok(None);
        };
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
    }

    let capacity = subject.owner.capacity;
    let width = bits_of_capacity(capacity) as i128;
    // emitter: one `and` per position with two wire inputs; the rank chain
    // is `lin` at no gate; the thermometer of a `Packed`'s ℓ, for `valid`.
    let wired = super::wired_positions(builder, value, &subject.owner, subject.valid);
    candidate = candidate.emit(Primitive::And, wired);
    if subject.valid && super::validity_is_a_thermometer(&subject.owner) {
        candidate = candidate.emit(Primitive::Therm { capacity }, 1);
    }
    candidate = candidate
        .output(
            Target::Value(value.clone()),
            Interface::Ranks { subject: subject.clone(), capacity, width },
            condition,
        )
        .parameter("B", capacity)
        .parameter("w", width)
        .parameter("B_v", wired);

    // `count.md`: the second bundle of the same wires, `r_B`, for the count
    // variable of Φ. It is absent in variant `valid`, and absent when the
    // mask was never used as a selector, so that Φ holds no count for it.
    if !subject.valid {
        if let Some(count) = builder.count_of(value) {
            let target = builder.wire_target(count);
            candidate = candidate.output(target, Interface::Wire(Dtype::Idx), condition);
        }
    }
    Ok(Some(candidate))
}
