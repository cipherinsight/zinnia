//! `PACK_BY_KEY` (`revision/phase1/rules/representation/pack_by_key.md`).
//!
//! Moves the valid elements of a bundle to the first positions of a new
//! payload, in key order, so that consumers scan a capacity sized by the
//! value's own length instead of the capacity of what it was selected from.
//! The key of a position is the prefix rank of the validity wires (layout
//! ranked) or the key wire (keyed); a compact provider is already packed and
//! is not matched.
//!
//! Two routing networks, chosen by `Ω.routing`: `select` routes each
//! position by a one-hot of its key, and answers the plain and the canonical
//! `Packed` from one node because the selection sum is 0 beyond ℓ;
//! `permute` gives every position a key of a permutation of `[0, B_in)` and
//! checks one `perm(B_in, 1 + S)`, with `permute-canon` zeroing the tail.
//!
//! **What is matched.** A rank-1 value with a `Masked` or ranked or keyed
//! `Store` provider, and — variant `rows` — a rank-≥2 ranked `Store` whose
//! trailing axes are constant and full, whose positions are its `B_0` rows.
//! The flat form of a rank-≥2 value held in a box with a run-time trailing
//! axis is not offered: its ℓ is the product of the run-time extents, for
//! which the group README's Open list has no port.

use crate::phase0::desc::ValueKey;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Capacity, Family, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::omega::Routing;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct PackByKey;

/// The families the `in` port may be filled from.
const INPUTS: [Family; 2] = [Family::Masked, Family::Store];

impl Rule for PackByKey {
    fn id(&self) -> RuleId {
        RuleId::PackByKey
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Packed, Family::Store]
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
        let Some(array) = builder.array(&key).cloned() else {
            return Ok(Vec::new());
        };
        if array.rank() == 0 || !super::super::sources::encoded(builder, "PACK_BY_KEY", array.dtype)
        {
            return Ok(Vec::new());
        }

        // resource_relations: B is the CONST answer, else MAX dims_v[0] over
        // Φ_e ∧ g, with the later-epoch variant of §6.2; the clamp by B_in
        // is a tightening, not a question.
        let mut probe = Candidate::new(RuleId::PackByKey, None, request.epoch);
        let under = builder.under(request.condition);
        let boxes = builder.capacity_box(&mut probe, &array.dims[..1], request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);

        let mut out = Vec::new();
        for family in INPUTS {
            // The `rows` output is a box `Store` and the ranked input it
            // packs is a `Store` too, so the input port may name the row
            // being filled. Asking for it would be the cut of §4 and would
            // make every `Store` row of a value provisional; the node is
            // offered from the `Packed` row instead, which carries the box
            // bundle as its second output (§3.4).
            if builder.table.is_filling_target(&request.target, family) {
                continue;
            }
            let row = builder.child(request, request.target.clone(), family, request.condition)?;
            for entry in &row.entries.clone() {
                let Some(lambda) = entry.interface.lambda().cloned() else {
                    continue;
                };
                // A compact provider is already packed; a box one would need
                // the product of its run-time extents as ℓ (README Open).
                let positions = match &lambda.layout {
                    LayoutKind::Ranked { capacity, .. } | LayoutKind::Keyed { capacity } => {
                        *capacity
                    }
                    _ => continue,
                };
                // `rows`: a rank-≥2 ranked Store over its C-order box; the
                // positions are the rows and S the slots per row.
                let (rows, per_row) = if array.rank() == 1 {
                    (positions, 1)
                } else {
                    let Some(shape) = self.row_shape(builder, &array, positions) else {
                        continue;
                    };
                    shape
                };
                for capacities in &boxes {
                    let capacity = capacities.axes[0].min(rows);
                    if request.family == Family::Store && array.rank() == 1 {
                        continue;
                    }
                    let mut candidate = Candidate::new(RuleId::PackByKey, None, capacities.epoch);
                    candidate.answers = probe.answers.clone();
                    candidate.matched.push(key.clone());
                    candidate.epoch = candidate.epoch.max(builder.watermark(&key));
                    candidate.inputs.push(entry.requirement);
                    candidate.epoch = candidate.epoch.max(entry.epoch);
                    // `len`: a `Masked` carries its own ℓ; every other input
                    // needs the extent on a wire.
                    if !matches!(entry.interface, Interface::Masked { .. })
                        && builder.terms.constant_value(array.dims[0]).is_none()
                    {
                        let length = builder.child(
                            request,
                            Target::Term(array.dims[0]),
                            Family::Wire,
                            request.condition,
                        )?;
                        let Some(first) = length.entries.first() else {
                            continue;
                        };
                        candidate.inputs.push(first.requirement);
                        candidate.epoch = candidate.epoch.max(first.epoch);
                    }
                    // `cond`: the gate of §5, because B comes from a MAX.
                    if gated {
                        let gate = builder.child(
                            request,
                            Target::BoolTerm(request.condition),
                            Family::Wire,
                            builder.terms.tt(),
                        )?;
                        let Some(first) = gate.entries.first() else {
                            continue;
                        };
                        candidate.inputs.push(first.requirement);
                        candidate.epoch = candidate.epoch.max(first.epoch);
                    }
                    out.extend(self.networks(
                        builder, request, &key, &array, candidate, capacity, rows, per_row, gated,
                    ));
                }
            }
        }
        Ok(out)
    }
}

impl PackByKey {
    /// The nodes the admitted networks give, with their bundles.
    #[allow(clippy::too_many_arguments)]
    fn networks(
        &self,
        builder: &Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        candidate: Candidate,
        capacity: Capacity,
        positions: Capacity,
        per_row: Capacity,
        gated: bool,
    ) -> Vec<Candidate> {
        let slots = capacity * per_row;
        let lambda = Layout::dense(LayoutKind::Compact(slots), key.clone());
        let packed = |canon: bool| Interface::Packed {
            dtype: array.dtype,
            capacity: slots,
            lambda: lambda.clone(),
            canon,
        };
        let gate = |candidate: Candidate| -> Candidate {
            if gated {
                candidate
                    .emit(Primitive::And, positions)
                    .emit(Primitive::Mul, 1)
            } else {
                candidate
            }
        };
        let numbered = |candidate: Candidate| -> Candidate {
            candidate
                .parameter("B", capacity)
                .parameter("B_in", positions)
                .parameter("S", per_row)
                .parameter("w", bits_of_capacity(positions) as i128)
        };
        let mut out = Vec::new();
        match builder.omega.routing {
            Routing::Select => {
                // `select`: one `onehot` per position and one `mul` per
                // (position, slot); the sum is empty beyond ℓ, so the
                // canonical bundle is the same wires (§3.4).
                let mut node = numbered(gate(candidate.clone()))
                    .emit(Primitive::OneHot { capacity }, positions)
                    .emit(Primitive::Mul, positions * capacity * per_row);
                node.variant = Some("select");
                let node = node
                    .output(request.target.clone(), packed(false), request.condition)
                    .output(request.target.clone(), packed(true), request.condition);
                out.push(self.rows(builder, request, key, array, node, capacity, per_row));
            }
            Routing::Permute => {
                // `permute`: the keys are a permutation of [0, B_in) and one
                // `perm(B_in, 1 + S)` carries the values across.
                let mut node = numbered(gate(candidate.clone()))
                    .emit(Primitive::Select, positions)
                    .emit(
                        Primitive::Perm { capacity: positions, arity: 1 + per_row as u32 },
                        1,
                    );
                node.variant = Some("permute");
                let canonical = node
                    .clone()
                    .emit(Primitive::Therm { capacity }, 1)
                    .emit(Primitive::Mul, capacity * per_row)
                    .output(request.target.clone(), packed(true), request.condition);
                let mut canonical = canonical;
                canonical.variant = Some("permute-canon");
                let node = node.output(request.target.clone(), packed(false), request.condition);
                out.push(self.rows(builder, request, key, array, node, capacity, per_row));
                out.push(self.rows(builder, request, key, array, canonical, capacity, per_row));
            }
        }
        out
    }

    /// The `rows` bundle: the box `Store` of the packed rows, whose validity
    /// is the thermometer of ℓ′ over B (`pack_by_key.md`, `rows`).
    #[allow(clippy::too_many_arguments)]
    fn rows(
        &self,
        builder: &Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        candidate: Candidate,
        capacity: Capacity,
        per_row: Capacity,
    ) -> Candidate {
        if array.rank() == 1 {
            return candidate;
        }
        let mut grid = vec![capacity];
        for &extent in &array.dims[1..] {
            grid.push(builder.terms.constant_value(extent).unwrap_or(0));
        }
        candidate
            .emit(Primitive::Therm { capacity }, 1)
            .output(
                Target::Value(key.clone()),
                Interface::Store(StoreIface {
                    owner: key.clone(),
                    dtype: array.dtype,
                    capacity: capacity * per_row,
                    lambda: Layout::dense(LayoutKind::Box(grid), key.clone()),
                }),
                request.condition,
            )
    }

    /// `rows`: the rows of a rank-≥2 ranked bundle and the slots per row,
    /// `None` when a trailing axis is not constant and full, where the
    /// prefix rank of a valid slot is not the C-order key of v.
    fn row_shape(
        &self,
        builder: &Builder<'_>,
        array: &crate::phase0::desc::ArrayDesc,
        positions: Capacity,
    ) -> Option<(Capacity, Capacity)> {
        let mut per_row: Capacity = 1;
        for &extent in &array.dims[1..] {
            per_row *= builder.terms.constant_value(extent)?;
        }
        if per_row <= 0 || positions % per_row != 0 {
            return None;
        }
        Some((positions / per_row, per_row))
    }
}
