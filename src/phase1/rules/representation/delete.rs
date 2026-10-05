//! `DELETE` (`revision/phase1/rules/representation/delete.md`).
//!
//! Represents `v = np.delete(a, obj, axis=k)` for one removed position. The
//! slots of v are the slots of a with the removed line skipped: slot p takes
//! a's entry p while `p_k` is below the position and a's entry `p + e_k`
//! from the position on. A constant position decides that pick at compile
//! time and the node costs nothing; a run-time position buys one thermometer
//! over the axis and one select per slot.

use crate::phase0::rules::Rule as Phase0Rule;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, Capacity, Family, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

use super::Position;

pub(crate) struct Delete;

impl Rule for Delete {
    fn id(&self) -> RuleId {
        RuleId::Delete
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Store, Family::Packed]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        if !matches!(request.family, Family::Store | Family::Packed) {
            return Ok(Vec::new());
        }
        let Some(key) = request.target.as_value().cloned() else {
            return Ok(Vec::new());
        };
        let Some(array) = builder.array(&key).cloned() else {
            return Ok(Vec::new());
        };
        let Some(matched) = super::axis_shift(builder, &key, Phase0Rule::EDelete, &array) else {
            return Ok(Vec::new());
        };
        if !super::super::sources::encoded(builder, "DELETE", array.dtype) {
            return Ok(Vec::new());
        }
        let Some(source) = builder.array(&matched.source).cloned() else {
            return Ok(Vec::new());
        };

        let mut probe = Candidate::new(RuleId::Delete, None, request.epoch);
        let under = builder.under(request.condition);
        // P1: the axis loses exactly one position, and the position itself.
        let Some(position) =
            super::position(builder, &mut probe, request, &matched, &source, -1)?
        else {
            return Ok(Vec::new());
        };
        // resource_relations: the capacity box of the *source*, whose grid
        // the `src` batch runs over; the output box is it with the axis one
        // shorter, because every other axis carries the same extent term.
        let boxes = builder.capacity_box(&mut probe, &source.dims, request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);

        let mut out = Vec::new();
        for capacities in boxes {
            let mut axes = capacities.axes.clone();
            if axes[matched.axis] == 0 {
                continue;
            }
            axes[matched.axis] -= 1;
            let slots: Capacity = axes.iter().product();
            let mut candidate =
                Candidate::new(RuleId::Delete, Some(position.variant()), capacities.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched = vec![key.clone(), matched.source.clone()];

            // `src`: every slot of a's capacity box, which holds both the
            // entry p and the entry p + e_k of every output slot p.
            let batch = Batch::grid(&capacities.axes);
            let values = builder.fetch_port(&mut candidate, request, &matched.source, &batch)?;
            let Some(first) = values.entries.first().cloned() else {
                continue;
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);

            if !super::attach_position(
                builder, &mut candidate, request, &matched, &source, &position,
            )? {
                continue;
            }
            if !super::attach_shapes(
                builder,
                &mut candidate,
                request,
                &array.dims,
                &capacities.constant,
                matched.axis,
            )? {
                continue;
            }
            if gated && !super::attach_condition(builder, &mut candidate, request)? {
                continue;
            }

            // emitter: a literal position picks each slot's source entry at
            // compile time; a run-time one needs the thermometer of the axis
            // and one select per slot.
            if matches!(position, Position::Wire) {
                candidate = candidate
                    .emit(
                        Primitive::Therm { capacity: capacities.axes[matched.axis] },
                        1,
                    )
                    .emit(super::value_gate(array.dtype), slots);
            }
            candidate = candidate
                .output(
                    Target::Value(key.clone()),
                    Interface::Store(StoreIface {
                        owner: key.clone(),
                        dtype: array.dtype,
                        capacity: slots,
                        lambda: Layout::dense(LayoutKind::Box(axes.clone()), key.clone()),
                    }),
                    request.condition,
                )
                .parameter("B", slots)
                .parameter("w", bits_of_capacity(slots) as i128)
                .parameter("axis", matched.axis as i128);
            if let Some(at) = position.literal() {
                candidate = candidate.parameter("pos", at);
            }
            for (axis, &capacity) in axes.iter().enumerate() {
                candidate = candidate.parameter(&format!("B_{}", axis), capacity);
            }
            // A rank-1 box in C-order is a `Packed` on the same wires (§3.4).
            if axes.len() == 1 {
                candidate = candidate.output(
                    Target::Value(key.clone()),
                    Interface::Packed {
                        dtype: array.dtype,
                        capacity: slots,
                        lambda: Layout::dense(LayoutKind::Compact(slots), key.clone()),
                        canon: false,
                    },
                    request.condition,
                );
            }
            out.push(candidate);
        }
        Ok(out)
    }
}
