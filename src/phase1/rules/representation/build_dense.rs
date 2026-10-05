//! `BUILD_DENSE` (`revision/phase1/rules/representation/build_dense.md`).
//!
//! Materialises a value: it takes the results of one read of every position
//! of a capacity box and declares them a `Store` in box layout, with the
//! batch's active flags as validity and the dimension terms on the shape
//! wires. It applies to any array value of Δ that is not an input, whatever
//! computes the batch, and it is the node that turns a computed batch into a
//! bundle other rules hold, filter, pack or read by key. It emits no gate:
//! the cost is in the `Fetch` it requires.
//!
//! A value consumed once needs no `BUILD_DENSE`, because its consumer takes
//! the batch directly; phase 2 decides that, by comparing this node's
//! subgraph with the consumer's own.

use crate::phase0::source::Producer;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, Family, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct BuildDense;

impl Rule for BuildDense {
    fn id(&self) -> RuleId {
        RuleId::BuildDense
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Store, Family::Packed]
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
        let Some(entry) = builder.entry(&key) else {
            return Ok(Vec::new());
        };
        // source_pattern: any array value whose producer is not `Input`, an
        // alias being matched through the value it names.
        if matches!(entry.producer, Producer::Input) || entry.producer.relabels().is_some() {
            return Ok(Vec::new());
        }
        let Some(array) = entry.desc.as_array() else {
            return Ok(Vec::new());
        };
        if array.rank() == 0 {
            return Ok(Vec::new());
        }
        let (dtype, dims) = (array.dtype, array.dims.clone());
        if !super::super::sources::encoded(builder, "BUILD_DENSE", dtype) {
            return Ok(Vec::new());
        }

        // premises and resource_relations: one CONST question per axis, and
        // a MAX where CONST fails, with the later-epoch variant of §6.2.
        let mut probe = Candidate::new(RuleId::BuildDense, None, request.epoch);
        let under = builder.under(request.condition);
        let boxes = builder.capacity_box(&mut probe, &dims, request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);

        let mut out = Vec::new();
        for capacities in boxes {
            let batch = Batch::grid(&capacities.axes);
            let mut candidate =
                Candidate::new(RuleId::BuildDense, Some("plain"), capacities.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched.push(key.clone());
            // `vals`: the grid batch of the capacity box. A provider of this
            // very row answering it would be "read v to rebuild v", which the
            // in-progress cut of §4 drops.
            let values = builder.fetch_port(&mut candidate, request, &key, &batch)?;
            let Some(values) = values.entries.first().cloned() else {
                continue;
            };
            candidate.inputs.push(values.requirement);
            candidate.epoch = candidate.epoch.max(values.epoch);
            // `shape_k`: one per axis on which CONST failed. The port is
            // required — the bundle states the extent on that wire — so an
            // axis with no provider drops the node rather than the port.
            let mut shapes = true;
            for (axis, &constant) in capacities.constant.iter().enumerate() {
                if constant {
                    continue;
                }
                let row = builder.child(
                    request,
                    Target::Term(dims[axis]),
                    Family::Wire,
                    request.condition,
                )?;
                let Some(first) = row.entries.first() else {
                    shapes = false;
                    break;
                };
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
            }
            if !shapes {
                continue;
            }
            // `cond`: the gate of §5, because Q comes from a MAX under g.
            if gated {
                let row = builder.child(
                    request,
                    Target::BoolTerm(request.condition),
                    Family::Wire,
                    builder.terms.tt(),
                )?;
                let Some(first) = row.entries.first() else {
                    continue;
                };
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
            }

            let slots = capacities.slots();
            let lambda = Layout::dense(LayoutKind::Box(capacities.axes.clone()), key.clone());
            candidate = candidate
                .output(
                    request.target.clone(),
                    Interface::Store(StoreIface {
                        owner: key.clone(),
                        dtype,
                        capacity: slots,
                        lambda,
                    }),
                    request.condition,
                )
                .parameter("Q", slots)
                .parameter("w", bits_of_capacity(slots) as i128);
            for (axis, &capacity) in capacities.axes.iter().enumerate() {
                candidate = candidate.parameter(&format!("B_{}", axis), capacity);
            }
            // The `Packed` attachment of §3.4: admitted when every axis
            // k ≥ 1 is CONST with B_k = dims_v[k], so that the C-order of the
            // box is the C-order of v.
            let compact = self.trailing_full(builder, &mut candidate, request, &dims, &capacities);
            if compact {
                let lambda_c = Layout::dense(LayoutKind::Compact(slots), key.clone());
                candidate = candidate.output(
                    request.target.clone(),
                    Interface::Packed {
                        dtype,
                        capacity: slots,
                        lambda: lambda_c,
                        canon: false,
                    },
                    request.condition,
                );
                // ℓ = g̃ · S · shape_0 is a `lin`, and one `mul` when gated.
                if gated {
                    candidate = candidate.emit(Primitive::Mul, 1);
                }
            }
            out.push(candidate.clone());

            // `canon`: the same node with `data_p = act_p · values_p`, which
            // is a third bundle on one set of wires (§3.4).
            if compact {
                let mut canonical = candidate;
                canonical.variant = Some("canon");
                canonical = canonical
                    .emit(Primitive::Mul, slots)
                    .output(
                        request.target.clone(),
                        Interface::Packed {
                            dtype,
                            capacity: slots,
                            lambda: Layout::dense(LayoutKind::Compact(slots), key.clone()),
                            canon: true,
                        },
                        request.condition,
                    );
                out.push(canonical);
            }
        }
        Ok(out)
    }
}

impl BuildDense {
    /// Whether every axis `k ≥ 1` is CONST with `B_k = dims_v[k]`, which is
    /// what makes the C-order of the box the C-order of v
    /// (`build_dense.md`, `produced_output_bundle`).
    fn trailing_full(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        request: &RowRequest,
        dims: &[crate::phase0::term::TermId],
        capacities: &crate::phase1::build::CapacityBox,
    ) -> bool {
        let under = builder.under(request.condition);
        for axis in 1..dims.len() {
            let answer = builder.ask_for(
                candidate,
                Question::Const(dims[axis]),
                capacities.epoch,
                under,
            );
            match answer {
                Verdict::Constant(extent) if extent == capacities.axes[axis] => {}
                _ => return false,
            }
        }
        true
    }
}
