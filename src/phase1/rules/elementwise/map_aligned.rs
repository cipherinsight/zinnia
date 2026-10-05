//! `MAP_ALIGNED` (`revision/phase1/rules/elementwise/map_aligned.md`).
//!
//! Applies one scalar primitive of one, two or three operands at every
//! logical position of v, over array operands whose bundles are *aligned*
//! (`rules/README.md` §4.1): every operand holds the same logical position in
//! the same physical slot. The primitive is emitted slot by slot, no position
//! moves, and the output is held in the family and layout of the operands
//! with their capacity; its validity, key, shape and length wires are the
//! operands' own wires by alias.
//!
//! Alignment is decided by construction except for the per-axis
//! "PROVE(dims equal) under g", which [`Builder::aligned`] asks.

use crate::phase1::build::Builder;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Family, Interface, Layout, LayoutKind, StoreIface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct MapAligned;

impl Rule for MapAligned {
    fn id(&self) -> RuleId {
        RuleId::MapAligned
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Store, Family::Masked, Family::Packed]
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
        // S3: every array port is in the family of the row. `Masked` needs
        // rank 1, which its own contract states.
        if request.family == Family::Masked && matched.dims.len() != 1 {
            return Ok(Vec::new());
        }
        // One port per array operand, in the row's family, and one per
        // scalar operand.
        let mut probe = Candidate::new(RuleId::MapAligned, None, request.epoch);
        let under = builder.under(request.condition);
        super::resolve_exponent(builder, &mut probe, &mut matched, request.epoch, under);
        // S4: the primitive has a row in the table.
        let Some(gadget) = super::gadget(builder, &matched) else {
            builder.missing.push(format!(
                "MAP_ALIGNED has no row for the primitive `{}` on {}",
                matched.primitive,
                matched.dtype.as_str()
            ));
            return Ok(Vec::new());
        };

        let mut rows = Vec::with_capacity(matched.arrays.len());
        for operand in &matched.arrays {
            let psi = builder.psi(operand);
            let condition =
                builder.child_under_g(&mut probe, request, psi, &operand.encode())?;
            let row = builder.child(
                request,
                Target::Value(operand.clone()),
                request.family,
                condition,
            )?;
            if row.entries.is_empty() {
                return Ok(Vec::new());
            }
            rows.push(row);
        }
        let mut scalar_ports = Vec::new();
        for operand in &matched.scalars {
            let psi = builder.psi(operand);
            let condition =
                builder.child_under_g(&mut probe, request, psi, &operand.encode())?;
            let row = builder.child(request, Target::Value(operand.clone()), Family::Wire, condition)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            scalar_ports.push((first.requirement, first.epoch));
        }

        // One node per combination the rows offer that satisfies S3 and the
        // alignment premise. The first port fixes the bundle; every other one
        // has to match it, so the combinations are the first row's entries
        // times the matching entries of the rest.
        let mut out = Vec::new();
        for first in &rows[0].entries {
            let Some(lambda) = first.interface.lambda().cloned() else {
                continue;
            };
            let capacity = capacity_of(&first.interface);
            let mut candidate = Candidate::new(RuleId::MapAligned, None, request.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched = std::iter::once(key.clone())
                .chain(matched.arrays.iter().cloned())
                .collect();
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
            let mut aligned = true;
            let own_dims = builder
                .array(&matched.arrays[0])
                .map(|array| array.dims.clone())
                .unwrap_or_default();
            for (at, row) in rows.iter().enumerate().skip(1) {
                let operand_dims = builder
                    .array(&matched.arrays[at])
                    .map(|array| array.dims.clone())
                    .unwrap_or_default();
                let mut matching = None;
                for entry in &row.entries {
                    let Some(other) = entry.interface.lambda().cloned() else {
                        continue;
                    };
                    if capacity_of(&entry.interface) != capacity {
                        continue;
                    }
                    // The alignment premise of §4.1, with its per-axis
                    // PROVE(dims equal) asked under the row's g.
                    if builder.aligned(
                        &mut candidate,
                        &lambda,
                        &other,
                        &own_dims,
                        &operand_dims,
                        request.epoch,
                        under,
                    ) {
                        matching = Some((entry.requirement, entry.epoch));
                        break;
                    }
                }
                match matching {
                    Some((requirement, epoch)) => {
                        candidate.inputs.push(requirement);
                        candidate.epoch = candidate.epoch.max(epoch);
                    }
                    None => {
                        aligned = false;
                        break;
                    }
                }
            }
            if !aligned {
                continue;
            }
            for &(requirement, epoch) in &scalar_ports {
                candidate.inputs.push(requirement);
                candidate.epoch = candidate.epoch.max(epoch);
            }
            // emitter: the same gadget at every slot, valid or not.
            for &(primitive, times) in &gadget {
                candidate = candidate.emit(primitive, times * capacity);
            }
            let lambda_v = super::output_layout(&lambda, &key);
            let interface = match request.family {
                Family::Store => Interface::Store(StoreIface {
                    owner: key.clone(),
                    dtype: matched.dtype,
                    capacity,
                    lambda: lambda_v.clone(),
                }),
                Family::Masked => Interface::Masked {
                    dtype: matched.dtype,
                    capacity,
                    lambda: lambda_v.clone(),
                },
                Family::Packed => Interface::Packed {
                    dtype: matched.dtype,
                    capacity,
                    lambda: lambda_v.clone(),
                    canon: false,
                },
                _ => continue,
            };
            candidate = candidate
                .output(request.target.clone(), interface, request.condition)
                .parameter("B", capacity);
            // §3.4: a rank-1 box `Store` is also a `Packed` over the same
            // wires, `data = values` and `ℓ = shape[0]`.
            if request.family == Family::Store
                && matched.dims.len() == 1
                && matches!(lambda_v.layout, LayoutKind::Box(_))
            {
                candidate = candidate.output(
                    request.target.clone(),
                    Interface::Packed {
                        dtype: matched.dtype,
                        capacity,
                        lambda: Layout::dense(LayoutKind::Compact(capacity), key.clone()),
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

/// `B`, the capacity of one bundle.
fn capacity_of(interface: &Interface) -> crate::phase1::interface::Capacity {
    match interface {
        Interface::Store(store) => store.capacity,
        Interface::Masked { capacity, .. } | Interface::Packed { capacity, .. } => *capacity,
        _ => 0,
    }
}
