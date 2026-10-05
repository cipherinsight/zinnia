//! `INSERT` (`revision/phase1/rules/representation/insert.md`).
//!
//! Represents `v = np.insert(a, obj, values, axis=k)` for one inserted
//! position. Slot p of v takes a's entry p while `p_k` is below the
//! position, the inserted element at the position, and a's entry `p − e_k`
//! above it. A constant position decides that pick at compile time and the
//! node costs nothing; a run-time position buys the thermometer and the
//! one-hot of the position over the axis and two selects per slot.

use crate::phase0::desc::Desc;
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

pub(crate) struct Insert;

impl Rule for Insert {
    fn id(&self) -> RuleId {
        RuleId::Insert
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
        let Some(matched) = super::axis_shift(builder, &key, Phase0Rule::EInsert, &array) else {
            return Ok(Vec::new());
        };
        if !super::super::sources::encoded(builder, "INSERT", array.dtype) {
            return Ok(Vec::new());
        }
        let Some(source) = builder.array(&matched.source).cloned() else {
            return Ok(Vec::new());
        };
        // `values`, the inserted element: the operand after the position.
        let Some(inserted) = matched.rest.first().cloned() else {
            return Ok(Vec::new());
        };
        // Covered: a scalar written on the whole inserted line, and an array
        // of the source's shape without the axis, one element per slot of
        // the line. Anything else is `insert.md`'s Open.
        let element_rank = match builder.entry(&inserted).map(|entry| &entry.desc) {
            Some(Desc::Scalar(_)) => 0,
            Some(Desc::Array(element)) if element.rank() == source.rank() - 1 => element.rank(),
            _ => return Ok(Vec::new()),
        };
        let element_dtype = match builder.entry(&inserted).map(|entry| &entry.desc) {
            Some(Desc::Scalar(scalar)) => scalar.dtype,
            Some(Desc::Array(element)) => element.dtype,
            _ => return Ok(Vec::new()),
        };

        let mut probe = Candidate::new(RuleId::Insert, None, request.epoch);
        let under = builder.under(request.condition);
        // P1: the axis gains exactly one position, and the position itself.
        let Some(position) = super::position(builder, &mut probe, request, &matched, &source, 1)?
        else {
            return Ok(Vec::new());
        };
        // resource_relations: the capacity box of the *source*, whose grid
        // the `src` batch runs over; the output box is it with the axis one
        // longer.
        let boxes = builder.capacity_box(&mut probe, &source.dims, request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);

        let mut out = Vec::new();
        for capacities in boxes {
            let mut axes = capacities.axes.clone();
            axes[matched.axis] += 1;
            let slots: Capacity = axes.iter().product();
            if slots > builder.omega.max_capacity {
                continue;
            }
            let mut candidate =
                Candidate::new(RuleId::Insert, Some(position.variant()), capacities.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched = vec![key.clone(), matched.source.clone(), inserted.clone()];

            // `src`: every slot of a's capacity box, which holds the entry
            // p − e_k of every output slot above the position and the entry
            // p of every output slot below it.
            let batch = Batch::grid(&capacities.axes);
            let values = builder.fetch_port(&mut candidate, request, &matched.source, &batch)?;
            let Some(first) = values.entries.first().cloned() else {
                continue;
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);

            // `val`: the inserted element, one wire for a scalar and one
            // entry per slot of the line for an array.
            let mut line = capacities.axes.clone();
            line.remove(matched.axis);
            let element_batch = if element_rank == 0 {
                Batch::Scalar
            } else {
                Batch::grid(&line)
            };
            let element = builder.fetch_port(&mut candidate, request, &inserted, &element_batch)?;
            let Some(first) = element.entries.first().cloned() else {
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

            // emitter: the cast of an inserted element whose dtype is not
            // the array's, once per entry of `val`.
            let line_slots: Capacity = line.iter().product();
            if let Some(cast) = super::cast_rows(element_dtype, array.dtype) {
                candidate = candidate.emit(
                    cast,
                    if element_rank == 0 { 1 } else { line_slots },
                );
            }
            // A run-time position needs the thermometer and the one-hot of
            // the position over the axis and two selects per slot, one fewer
            // on the first and on the last position of the axis.
            if matches!(position, Position::Wire) {
                candidate = candidate
                    .emit(Primitive::Therm { capacity: axes[matched.axis] }, 1)
                    .emit(Primitive::OneHot { capacity: axes[matched.axis] }, 1)
                    .emit(
                        super::value_gate(array.dtype),
                        2 * capacities.axes[matched.axis] * line_slots,
                    );
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
