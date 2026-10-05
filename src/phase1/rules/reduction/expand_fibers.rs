//! `EXPAND_FIBERS` (`revision/phase1/rules/reduction/expand_fibers.md`).
//!
//! A reduction over the axes A of a value v combines, for every output
//! position, a fixed list of elements of v: its *fiber*. `EXPAND_FIBERS` lays
//! every fiber out as W slots with a value wire and a validity bit each, so
//! that `FOLD_GROUPS` and `CUMSUM` combine slots without knowing where v is
//! held. It is interface-driven: it matches any value of rank ≥ 1 that has a
//! provider in the Store, Masked or Packed family, or that can be fetched.
//!
//! The variants are `box`, `compact`, `ranked`, `keyed` and `fetch`;
//! `reshape` needs the `Groups` of the reshaped value, which `fetch`
//! answers directly here.
//!
//! **The validity of a `Packed`.** `compact` over a `Packed` needs the
//! thermometer of ℓ, which the bundle does not carry. `rules/README.md` §4.1
//! makes that thermometer the `Ranks(I, valid)` node of the bundle, built
//! once and shared by every consumer that holds it, so the variant raises it
//! as a port rather than emitting one of its own.

use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::interface::{
    Batch, Capacity, Family, GroupFlags, GroupsIface, Interface, Layout, LayoutKind,
};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ExpandFibers;

impl Rule for ExpandFibers {
    fn id(&self) -> RuleId {
        RuleId::ExpandFibers
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Groups]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let (Some(key), Some(axes)) = (request.target.as_value(), request.axes.clone()) else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        let Some(array) = builder.array(&key).cloned() else {
            return Ok(Vec::new());
        };
        if array.rank() == 0 || axes.is_empty() {
            return Ok(Vec::new());
        }
        let dims = array.dims.clone();
        let outer: Vec<usize> = (0..dims.len()).filter(|axis| !axes.contains(axis)).collect();
        let mut out = Vec::new();

        // The bundle variants: the slots of a provider of v.
        for family in [Family::Store, Family::Masked, Family::Packed] {
            let row = builder.child(request, request.target.clone(), family, request.condition)?;
            for entry in &row.entries.clone() {
                let Some(lambda) = entry.interface.lambda().cloned() else {
                    continue;
                };
                let Some((variant, fibers, width, flags)) =
                    self.bundle(&lambda, &entry.interface, &axes, &outer, dims.len())
                else {
                    continue;
                };
                let mut candidate =
                    Candidate::new(RuleId::ExpandFibers, Some(variant), request.epoch);
                candidate.matched.push(key.clone());
                candidate.inputs.push(entry.requirement);
                candidate.epoch = candidate.epoch.max(entry.epoch);
                // `compact` on a `Packed` needs the thermometer of ℓ: the
                // `Ranks(I, valid)` node of the same bundle (§4.1).
                if variant == "compact"
                    && matches!(entry.interface, Interface::Packed { .. })
                {
                    let Some(subject) =
                        crate::phase1::rules::masks::subject_of(&entry.interface, &key, true)
                    else {
                        continue;
                    };
                    let row = builder.ranks_port(&mut candidate, request, &key, subject)?;
                    let Some(first) = row.entries.first() else {
                        continue;
                    };
                    candidate.inputs.push(first.requirement);
                    candidate.epoch = candidate.epoch.max(first.epoch);
                }
                candidate = candidate.output(
                    request.target.clone(),
                    Interface::Groups(GroupsIface {
                        owner: key.clone(),
                        axes: axes.clone(),
                        fibers,
                        width,
                        dtype: array.dtype,
                        flags,
                        slots: lambda.clone(),
                    }),
                    request.condition,
                );
                candidate = candidate.parameter("P", fibers).parameter("W", width);
                out.push(candidate);
            }
        }

        // `fetch`: no representation of v is read; one `Fetch` over the
        // capacity grid, whose capacities are CONST or MAX per axis.
        let mut probe = Candidate::new(RuleId::ExpandFibers, Some("fetch"), request.epoch);
        let under = builder.under(request.condition);
        for capacities in builder.capacity_box(&mut probe, &dims, request.epoch, under) {
            let mut candidate =
                Candidate::new(RuleId::ExpandFibers, Some("fetch"), capacities.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched.push(key.clone());
            let batch = Batch::grid(&capacities.axes);
            let values = builder.fetch_port(&mut candidate, request, &key, &batch)?;
            let Some(values) = values.entries.first().cloned() else {
                continue;
            };
            candidate.inputs.push(values.requirement);
            candidate.epoch = candidate.epoch.max(values.epoch);
            let fibers: Capacity = outer.iter().map(|&axis| capacities.axes[axis]).product();
            let width: Capacity = axes.iter().map(|&axis| capacities.axes[axis]).product();
            // OV-ALIAS: PROVE(⋀_{a∈A} d_a ≥ 1) decides whether `ov` is the
            // validity of slot 0 by alias or a thermometer per outer axis.
            let one = builder.literal(1);
            let non_empty = axes
                .iter()
                .all(|&axis| builder.prove_le(&mut candidate, one, dims[axis], capacities.epoch, under));
            if !non_empty {
                for &axis in &outer {
                    if !capacities.constant[axis] {
                        candidate = candidate
                            .emit(Primitive::Therm { capacity: capacities.axes[axis] }, 1);
                    }
                }
            }
            let prefix = axes.len() == 1
                || axes.iter().skip(1).all(|&axis| capacities.constant[axis]);
            candidate = candidate
                .output(
                    request.target.clone(),
                    Interface::Groups(GroupsIface {
                        owner: key.clone(),
                        axes: axes.clone(),
                        fibers,
                        width,
                        dtype: array.dtype,
                        flags: GroupFlags { ordered: true, prefix, zeroed: false },
                        slots: Layout::dense(
                            LayoutKind::Box(capacities.axes.clone()),
                            key.clone(),
                        ),
                    }),
                    request.condition,
                )
                .parameter("P", fibers)
                .parameter("W", width);
            out.push(candidate);
        }
        Ok(out)
    }
}

impl ExpandFibers {
    /// Which variant a bundle gives, with its P, W and flags
    /// (`expand_fibers.md`, `produced_output_bundle`).
    fn bundle(
        &self,
        lambda: &Layout,
        interface: &Interface,
        axes: &[usize],
        outer: &[usize],
        rank: usize,
    ) -> Option<(&'static str, Capacity, Capacity, GroupFlags)> {
        let canon = matches!(interface, Interface::Packed { canon: true, .. });
        match &lambda.layout {
            // `box`: any A; the slots are the box slots.
            LayoutKind::Box(grid) => {
                if grid.len() != rank {
                    return None;
                }
                let fibers: Capacity = outer.iter().map(|&axis| grid[axis]).product();
                let width: Capacity = axes.iter().map(|&axis| grid[axis]).product();
                let prefix = axes.len() == 1;
                Some((
                    "box",
                    fibers,
                    width,
                    GroupFlags { ordered: true, prefix, zeroed: false },
                ))
            }
            // `compact`: A is every axis, so P = 1.
            LayoutKind::Compact(capacity) => {
                if axes.len() != rank {
                    return None;
                }
                Some((
                    "compact",
                    1,
                    *capacity,
                    GroupFlags { ordered: true, prefix: true, zeroed: canon },
                ))
            }
            // `ranked`: rank 1, A = {0}.
            LayoutKind::Ranked { capacity, .. } => {
                if rank != 1 || axes != [0] {
                    return None;
                }
                Some((
                    "ranked",
                    1,
                    *capacity,
                    GroupFlags { ordered: true, prefix: false, zeroed: false },
                ))
            }
            // `keyed`: rank 1, A = {0}; the only variant with ordered = no,
            // the slots carrying v's elements with their key wires.
            LayoutKind::Keyed { capacity } => {
                if rank != 1 || axes != [0] {
                    return None;
                }
                Some((
                    "keyed",
                    1,
                    *capacity,
                    GroupFlags { ordered: false, prefix: false, zeroed: false },
                ))
            }
        }
    }
}
