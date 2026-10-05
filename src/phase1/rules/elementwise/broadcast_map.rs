//! `BROADCAST_MAP` (`revision/phase1/rules/elementwise/broadcast_map.md`).
//!
//! Represents an elementwise result whose array operands are not aligned
//! because broadcasting between arrays is involved: some operand contributes
//! an axis of extent 1 or a smaller rank. It builds a dense box `Store` of v:
//! every operand is read at the coordinate broadcasting assigns to each slot,
//! by a `Fetch` over the grid, and the primitive of `map_aligned.md`'s table
//! is applied slot by slot. The capacity box is inherited from an operand
//! that is full on every axis and held in a box `Store` (variant F) or
//! allocated by one MAX per axis (variant G).

use crate::phase0::term::TermId;

use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, Capacity, Coordinate, Family, Interface, Layout, LayoutKind, RangeAxis, StoreIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::{Condition, Question};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct BroadcastMap;

/// What one operand contributes on one output axis (`broadcast_map.md`,
/// premises).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// The operand has no axis there: a broadcast axis with no coordinate.
    Absent,
    /// D1: the extents agree.
    Full,
    /// D2: the operand's extent is 1, so it is read at position 0.
    Broadcast,
}

impl Rule for BroadcastMap {
    fn id(&self) -> RuleId {
        RuleId::BroadcastMap
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Store]
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
        let Some(mut matched) = super::matches(builder, &key) else {
            return Ok(Vec::new());
        };
        // S1: at least two array operands, which is what makes broadcasting
        // between arrays possible at all.
        if matched.arrays.len() < 2 {
            return Ok(Vec::new());
        }
        let mut probe = Candidate::new(RuleId::BroadcastMap, None, request.epoch);
        let under = builder.under(request.condition);
        super::resolve_exponent(builder, &mut probe, &mut matched, request.epoch, under);
        let Some(gadget) = super::gadget(builder, &matched) else {
            return Ok(Vec::new());
        };

        // The per-axis decision for every operand.
        let mut decisions = Vec::with_capacity(matched.arrays.len());
        for operand in &matched.arrays {
            let Some(decision) =
                self.decide(builder, &mut probe, &matched.dims, operand, request.epoch, under)
            else {
                // D3: undecided. `Represent(v, Store)` is then `BUILD_DENSE`
                // over `ELEMENTWISE_BATCH`.
                return Ok(Vec::new());
            };
            decisions.push(decision);
        }
        // P0: some operand is absent or broadcast on some axis; otherwise the
        // operands are `MAP_ALIGNED`'s.
        if decisions
            .iter()
            .all(|decision| decision.iter().all(|axis| *axis == Axis::Full))
        {
            return Ok(Vec::new());
        }

        // Variant G: the capacity box from CONST or MAX per axis, with the
        // later-epoch variants of §6.2. Variant F would inherit a full
        // operand's box; it is the same box whenever that operand's extents
        // are the output's, which D1 established, so the box is taken here
        // once and the `full` port is the operand's own `Fetch`.
        let boxes = builder.capacity_box(&mut probe, &matched.dims, request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);
        let mut out = Vec::new();
        for capacities in boxes {
            let mut candidate = Candidate::new(RuleId::BroadcastMap, Some("G"), capacities.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched = std::iter::once(key.clone())
                .chain(matched.arrays.iter().cloned())
                .collect();
            let mut complete = true;
            for (at, operand) in matched.arrays.iter().enumerate() {
                let batch = self.operand_batch(&capacities.axes, &decisions[at]);
                let row = builder.fetch_port(&mut candidate, request, operand, &batch)?;
                let Some(first) = row.entries.first() else {
                    complete = false;
                    break;
                };
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
            }
            if !complete {
                continue;
            }
            for operand in &matched.scalars {
                let psi = builder.psi(operand);
                let condition =
                    builder.child_under_g(&mut candidate, request, psi, &operand.encode())?;
                let row = builder.child(
                    request,
                    Target::Value(operand.clone()),
                    Family::Wire,
                    condition,
                )?;
                let Some(first) = row.entries.first() else {
                    complete = false;
                    break;
                };
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
            }
            if !complete {
                continue;
            }
            // `shape_k`: one per axis whose extent is not CONST.
            for (axis, &constant) in capacities.constant.iter().enumerate() {
                if constant {
                    continue;
                }
                let row = builder.child(
                    request,
                    Target::Term(matched.dims[axis]),
                    Family::Wire,
                    request.condition,
                )?;
                if let Some(first) = row.entries.first() {
                    candidate.inputs.push(first.requirement);
                    candidate.epoch = candidate.epoch.max(first.epoch);
                }
            }
            let slots = capacities.slots();
            for (primitive, times) in gadget.clone() {
                candidate = candidate.emit(primitive, times * slots);
            }
            if gated {
                candidate = candidate.emit(Primitive::Mul, 0);
            }
            candidate = candidate
                .output(
                    request.target.clone(),
                    Interface::Store(StoreIface {
                        owner: key.clone(),
                        dtype: matched.dtype,
                        capacity: slots,
                        lambda: Layout::dense(
                            LayoutKind::Box(capacities.axes.clone()),
                            key.clone(),
                        ),
                    }),
                    request.condition,
                )
                .parameter("B", slots);
            // §3.4: a rank-1 box `Store` is also a `Packed` over the same
            // wires.
            if matched.dims.len() == 1 {
                candidate = candidate.output(
                    request.target.clone(),
                    Interface::Packed {
                        dtype: matched.dtype,
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

impl BroadcastMap {
    /// D1 to D3 of `broadcast_map.md` for one operand, per output axis.
    fn decide(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        dims: &[TermId],
        operand: &crate::phase0::desc::ValueKey,
        epoch: u32,
        under: Condition,
    ) -> Option<Vec<Axis>> {
        let operand_dims = builder.array(operand).map(|array| array.dims.clone())?;
        let offset = dims.len().checked_sub(operand_dims.len())?;
        let mut out = Vec::with_capacity(dims.len());
        for axis in 0..dims.len() {
            if axis < offset {
                out.push(Axis::Absent);
                continue;
            }
            let operand_axis = axis - offset;
            if builder.prove_eq(
                candidate,
                operand_dims[operand_axis],
                dims[axis],
                epoch,
                under,
            ) {
                out.push(Axis::Full);
                continue;
            }
            match builder.ask_for(
                candidate,
                Question::Const(operand_dims[operand_axis]),
                epoch,
                under,
            ) {
                Verdict::Constant(1) => out.push(Axis::Broadcast),
                _ => return None,
            }
        }
        Some(out)
    }

    /// `batch_j`: the grid batch with the per-axis template the decision
    /// prescribes — `(0, 1)` on a full axis, `(0, 0)` on a broadcast one, and
    /// no coordinate where the operand is absent.
    fn operand_batch(&self, capacities: &[Capacity], decision: &[Axis]) -> Batch {
        let mut axes = Vec::new();
        for (axis, &width) in capacities.iter().enumerate() {
            match decision.get(axis) {
                Some(Axis::Absent) | None => {}
                Some(Axis::Full) => axes.push(RangeAxis {
                    base: Coordinate::Literal(0),
                    stride: 1,
                    width,
                    broadcast: None,
                }),
                Some(Axis::Broadcast) => axes.push(RangeAxis {
                    base: Coordinate::Literal(0),
                    stride: 0,
                    width,
                    broadcast: None,
                }),
            }
        }
        Batch::Range { axes, live: crate::phase1::interface::Live::InRange }
    }
}
