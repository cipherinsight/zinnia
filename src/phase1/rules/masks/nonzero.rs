//! `NONZERO` (`revision/phase1/rules/masks/nonzero.md`).
//!
//! `nz = nonzero(m)` or `flatnonzero(m)`: the flat C-order indices of the
//! true elements of m, in increasing order, packed into the first positions
//! of a payload of capacity `B_nz = min(MAX c, B)`. Every physical position
//! i of an ordered bundle I of m carries its flat logical index `key_i`; the
//! active positions, whose ranks come from the `PREFIX_RANK` node over I,
//! are routed to slot `r_i`. The routing network is `PACK_BY_KEY`'s, with
//! `key_i` as the payload, and `Ω.routing` chooses it.
//!
//! **Where `key_i` comes from.** A compact bundle or a `Packed` holds
//! logical index i at slot i, so `key_i` is the constant i; a box bundle of
//! rank 1 likewise. A box bundle of rank ≥ 2 has `key_i = Σ_k p_k·stride_k`,
//! and a stride that is not CONST is a term port — built only where the
//! product of the trailing extents is a term of Φ, which it is when at most
//! one of them is run-time. A ranked or `Masked` bundle has no such key: its
//! slots are positions of a base, and the logical index of a valid slot is
//! the rank of the validity wires, which is the `Ranks(I, valid)` node of
//! the same bundle (`rules/README.md` §4.5) and makes the keys wires.

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Capacity, Family, Interface, Layout, LayoutKind, RanksOf,
};
use crate::phase1::omega::Routing;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Nonzero;

/// The families a bundle of the mask may be held in.
const BUNDLES: [Family; 3] = [Family::Store, Family::Masked, Family::Packed];

impl Rule for Nonzero {
    fn id(&self) -> RuleId {
        RuleId::Nonzero
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Packed]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(key) = request.target.as_value() else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        let Some(mask) = self.matches(builder, &key) else {
            return Ok(Vec::new());
        };
        // `np.nonzero(m)` is a *tuple* of one index array on a rank-1 mask
        // (`../../phase0/definition.md` §5.5), so the value this row is about
        // is a Composite and its extents are its one element's. A consumer
        // that wrote `m.nonzero()[0]` reaches this row through the
        // composite-index delegation of `../../build.rs`.
        let Some(dims) = builder
            .array(&key)
            .map(|array| array.dims.clone())
            .or_else(|| {
                let composite = builder.entry(&key)?.desc.as_composite()?;
                match composite.elems.as_slice() {
                    [crate::phase0::desc::Desc::Array(element)] => Some(element.dims.clone()),
                    _ => None,
                }
            })
        else {
            return Ok(Vec::new());
        };
        let mask_dims = builder
            .array(&mask)
            .map(|array| array.dims.clone())
            .unwrap_or_default();

        // premises: MAX c for the capacity, CONST g for the gate, with the
        // later-epoch variant of §6.2 where a check tightened c.
        let mut probe = Candidate::new(RuleId::Nonzero, None, request.epoch);
        let under = builder.under(request.condition);
        let boxes = builder.capacity_box(&mut probe, &dims, request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);

        let mut out = Vec::new();
        for family in BUNDLES {
            let row = builder.child(request, Target::Value(mask.clone()), family, request.condition)?;
            for entry in &row.entries.clone() {
                let Some(subject) = super::subject_of(&entry.interface, &mask, false) else {
                    continue;
                };
                for capacities in &boxes {
                    let capacity = capacities.axes[0].min(subject.owner.capacity);
                    let mut candidate = Candidate::new(
                        RuleId::Nonzero,
                        Some(match builder.omega.routing {
                            Routing::Select => "select",
                            Routing::Permute => "permute",
                        }),
                        capacities.epoch,
                    );
                    candidate.answers = probe.answers.clone();
                    candidate.matched.push(key.clone());
                    candidate.epoch = candidate.epoch.max(builder.watermark(&key));

                    // `ranks`: the `PREFIX_RANK` node over this bundle of m.
                    let ranks =
                        builder.ranks_port(&mut candidate, request, &mask, subject.clone())?;
                    let Some(first) = ranks.entries.first() else {
                        continue;
                    };
                    candidate.inputs.push(first.requirement);
                    candidate.epoch = candidate.epoch.max(first.epoch);

                    // `key_i`: a constant per slot, or the wires of the
                    // validity ranks over a ranked bundle.
                    let Some(wire_keys) = self.keys(
                        builder,
                        &mut candidate,
                        request,
                        &mask,
                        &subject,
                        &mask_dims,
                    )?
                    else {
                        continue;
                    };
                    // `cond`: the gate of §5, because B_nz comes from a MAX.
                    if gated {
                        let row = builder.child(
                            request,
                            Target::BoolTerm(request.condition),
                            Family::Wire,
                            builder.terms.tt(),
                        )?;
                        let Some(gate) = row.entries.first() else {
                            continue;
                        };
                        candidate.inputs.push(gate.requirement);
                        candidate.epoch = candidate.epoch.max(gate.epoch);
                    }

                    let positions = subject.owner.capacity;
                    let wires = i128::from(wire_keys);
                    candidate = match builder.omega.routing {
                        // `select`: one `onehot` per position, and one `mul`
                        // per (position, slot) when the keys are wires.
                        Routing::Select => candidate
                            .emit(Primitive::OneHot { capacity }, positions)
                            .emit(Primitive::Mul, wires * positions * capacity)
                            .emit(Primitive::And, i128::from(gated) * positions)
                            .emit(Primitive::Mul, i128::from(gated)),
                        // `permute`: the keys of a permutation of [0, B) and
                        // one `perm(B, 2)`.
                        Routing::Permute => candidate
                            .emit(Primitive::Select, positions)
                            .emit(Primitive::Perm { capacity: positions, arity: 2 }, 1)
                            .emit(Primitive::Mul, i128::from(gated)),
                    };
                    let lambda =
                        Layout::dense(LayoutKind::Compact(capacity), key.clone());
                    candidate = candidate
                        .output(
                            request.target.clone(),
                            Interface::Packed {
                                dtype: Dtype::Idx,
                                capacity,
                                lambda: lambda.clone(),
                                canon: false,
                            },
                            request.condition,
                        )
                        .parameter("B", subject.owner.capacity)
                        .parameter("B_nz", capacity)
                        .parameter("w", bits_of_capacity(subject.owner.capacity) as i128);
                    // `select` leaves every unrouted slot 0, so the canonical
                    // bundle is the same wires (§3.4); `permute` would need a
                    // thermometer and B products, which `Ω.routing` does not
                    // ask for here.
                    if matches!(builder.omega.routing, Routing::Select) {
                        candidate = candidate.output(
                            request.target.clone(),
                            Interface::Packed {
                                dtype: Dtype::Idx,
                                capacity,
                                lambda,
                                canon: true,
                            },
                            request.condition,
                        );
                    }
                    out.push(candidate);
                }
            }
        }
        Ok(out)
    }
}

impl Nonzero {
    /// `source_pattern`: `Applied([E-Nonzero], [m], ·)` after the alias
    /// links, with m the mask the count is keyed by.
    fn matches(&self, builder: &Builder<'_>, key: &ValueKey) -> Option<ValueKey> {
        let entry = builder.entry(key)?;
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return None;
        };
        if !rules.iter().all(|rule| *rule == Phase0Rule::ENonzero) || rules.is_empty() {
            return None;
        }
        let mask = operands
            .iter()
            .find(|operand| builder.array(operand).is_some())?;
        Some(builder.defining_key(mask))
    }

    /// The payload `key_i`, the flat logical index of position i.
    ///
    /// Returns whether the keys are wires, or `None` when the bundle offers
    /// no key the rule can write: a box bundle of rank ≥ 2 whose trailing
    /// extents do not multiply to a term of Φ.
    fn keys(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        request: &RowRequest,
        mask: &ValueKey,
        subject: &RanksOf,
        mask_dims: &[TermId],
    ) -> Result<Option<bool>, Reject> {
        match &subject.owner.layout {
            // Slot j holds logical index j: the key is the constant j.
            LayoutKind::Compact(_) => Ok(Some(false)),
            LayoutKind::Box(grid) => {
                if grid.len() <= 1 {
                    return Ok(Some(false));
                }
                // The strides run over the mask's own extents, so the box
                // must have one capacity per axis of it.
                if grid.len() != mask_dims.len() {
                    return Ok(None);
                }
                // `stride_k = ∏_{a>k} dims_m[a]`, one port per axis whose
                // stride is not CONST.
                let mut wires = false;
                for axis in 0..grid.len().saturating_sub(1) {
                    let Some(stride) = self.stride(builder, &mask_dims[axis + 1..]) else {
                        return Ok(None);
                    };
                    if builder.terms.constant_value(stride).is_some() {
                        continue;
                    }
                    wires = true;
                    let row = builder.child(
                        request,
                        Target::Term(stride),
                        Family::Wire,
                        request.condition,
                    )?;
                    let Some(first) = row.entries.first() else {
                        return Ok(None);
                    };
                    candidate.inputs.push(first.requirement);
                    candidate.epoch = candidate.epoch.max(first.epoch);
                }
                Ok(Some(wires))
            }
            // A ranked bundle's slots are a base's positions: the logical
            // index of a valid slot is the rank of the validity wires.
            LayoutKind::Ranked { .. } => {
                let validity = RanksOf { owner: subject.owner.clone(), valid: true };
                let row = builder.ranks_port(candidate, request, mask, validity)?;
                let Some(first) = row.entries.first() else {
                    return Ok(None);
                };
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
                Ok(Some(true))
            }
            LayoutKind::Keyed { .. } => Ok(None),
        }
    }

    /// `∏ factors` as a term of Φ: the product is in the term language only
    /// where at most one factor is not a literal (`../README.md` §6.3), so a
    /// box bundle with two run-time trailing extents has no key here.
    fn stride(&self, builder: &mut Builder<'_>, factors: &[TermId]) -> Option<TermId> {
        let mut constant: Capacity = 1;
        let mut runtime: Option<TermId> = None;
        for &factor in factors {
            match builder.terms.constant_value(factor) {
                Some(value) => constant *= value,
                None if runtime.is_none() => runtime = Some(factor),
                None => return None,
            }
        }
        Some(match runtime {
            Some(term) => builder.terms.mul_const(constant, term),
            None => builder.terms.constant(constant),
        })
    }
}
