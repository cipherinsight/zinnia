//! `CONCAT` (`revision/phase1/rules/representation/concat.md`).
//!
//! Represents `u = concatenate([v_1, …, v_n], axis)` or
//! `stack([v_1, …, v_n], axis)` without computing anything about the
//! elements, in one of three ways: by wiring, when every segment's extent on
//! the axis is constant, so every slot of u is a fixed slot of one segment;
//! by laying the segments' ordered bundles side by side with their validity
//! bits, which is already an in-place bundle of u along axis 0; or by the
//! compact payload of `PACK_SEGMENTS`.
//!
//! **Where the axis comes from.** The front end spells a concatenation six
//! ways — `concatenate`, `vstack`, `hstack`, `dstack`, `column_stack`,
//! `row_stack` — and phase 0 normalises each to one `E-Concat` with the axis
//! already applied. The axis is therefore read off Δ rather than off the AST
//! node: it is the one axis on which the output's extent term differs from
//! the first segment's, which is what `E-Concat` and `E-Stack` produce
//! (`../../../phase0/definition.md` §5.6).

use crate::phase0::desc::ValueKey;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;

use crate::phase1::build::{bits_of_capacity, Builder, CapacityBox};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, Capacity, Family, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Concat;

/// One segment of a join: the row it is read from, its rank, and its
/// extents.
///
/// A segment of rank ≥ 1 is a value of Δ read by a `Fetch`. A rank-0 segment
/// — `np.array([min(a.shape), n])`, the Boolean of each iteration of
/// `any(genexp)` — is one element and one wire, and Δ may name it by its
/// term rather than by a key of its own (`rules/README.md` §2), so the
/// target is what the row asks for and `key` is present only where the
/// element has one.
struct Segment {
    target: Target,
    key: Option<ValueKey>,
    rank: usize,
    dims: Vec<crate::phase0::term::TermId>,
}

/// What the source pattern read.
struct Matched {
    /// The Composite of the operands, which `PACK_SEGMENTS` owns.
    list: ValueKey,
    /// The segments, in order.
    segments: Vec<Segment>,
    /// The axis the segments are joined on.
    axis: usize,
    /// Whether a new axis of extent 1 per segment is inserted (`E-Stack`).
    stacked: bool,
}

impl Rule for Concat {
    fn id(&self) -> RuleId {
        RuleId::Concat
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
        let Some(matched) = self.matches(builder, &key) else {
            return Ok(Vec::new());
        };
        let Some(array) = builder.array(&key).cloned() else {
            return Ok(Vec::new());
        };
        if !super::super::sources::encoded(builder, "CONCAT", array.dtype) {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        out.extend(self.wired(builder, request, &key, &array, &matched)?);
        if matched.axis == 0 && !matched.stacked {
            out.extend(self.in_place(builder, request, &key, &array, &matched)?);
            out.extend(self.packed(builder, request, &key, &array, &matched)?);
        }
        Ok(out)
    }
}

impl Concat {
    /// `source_pattern`: `Applied(E-Concat | E-Stack, [L])` with L the
    /// `E-Composite` of the operands.
    fn matches(&self, builder: &Builder<'_>, key: &ValueKey) -> Option<Matched> {
        let entry = builder.entry(key)?;
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return None;
        };
        // A value that was joined *and* reduced is one key with two rules
        // and no key for the join's own result, so no rule of this group can
        // name what it would represent.
        if !rules
            .iter()
            .all(|rule| matches!(rule, Phase0Rule::EConcat | Phase0Rule::EStack))
            || rules.is_empty()
        {
            return None;
        }
        let list = operands.iter().find(|operand| {
            builder
                .entry(operand)
                .map(|entry| entry.desc.as_composite().is_some())
                .unwrap_or(false)
        })?;
        let composite = builder.entry(list)?.desc.as_composite()?;
        let mut segments = Vec::with_capacity(composite.elems.len());
        for at in 0..composite.elems.len() {
            let target = builder.element_target(list, &[at])?;
            // The element's own descriptor says its rank: a scalar is rank 0
            // and carries no extent, an array carries one per axis.
            let (rank, dims) = match builder.element_of(list, &[at])? {
                crate::phase0::desc::Desc::Array(element) => {
                    (element.rank(), element.dims.clone())
                }
                crate::phase0::desc::Desc::Scalar(_) => (0, Vec::new()),
                _ => return None,
            };
            let key = target.as_value().cloned();
            segments.push(Segment { target, key, rank, dims });
        }
        let array = builder.array(key)?;
        let first = segments.first()?;
        // `E-Stack` inserts an axis, `E-Concat` widens one: the joined axis
        // is the first on which the output's extent term differs from the
        // first segment's, and for a stack whose new axis is last there is
        // none, so it is the segment's rank.
        let stacked = array.rank() == first.rank + 1;
        let differing = (0..array.rank()).find(|&axis| first.dims.get(axis) != array.dims.get(axis));
        let axis = match (stacked, differing) {
            (true, Some(axis)) => axis,
            (true, None) => first.rank,
            (false, Some(axis)) if axis < first.rank => axis,
            // Neither shape says which axis was joined — the operands were
            // promoted to another rank before the join — so the value is no
            // node of this rule (`concat.md`, Open).
            _ => return None,
        };
        Some(Matched { list: list.clone(), segments, axis, stacked })
    }

    /// `wire`: every slot of u is a fixed slot of one segment, so the node
    /// allocates nothing and its ports are one `Fetch` per segment.
    fn wired(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        matched: &Matched,
    ) -> Result<Vec<Candidate>, Reject> {
        if !matches!(request.family, Family::Store | Family::Packed) {
            return Ok(Vec::new());
        }
        let mut probe = Candidate::new(RuleId::Concat, Some("wire"), request.epoch);
        let under = builder.under(request.condition);
        // premises: CONST dims_{v_k}[ax] for every segment; a run-time extent
        // on the joined axis makes the wiring undecidable and drops `wire`.
        let mut extents = Vec::with_capacity(matched.segments.len());
        for segment in &matched.segments {
            if matched.stacked {
                extents.push(1);
                continue;
            }
            let Some(&extent) = segment.dims.get(matched.axis) else {
                return Ok(Vec::new());
            };
            let answer =
                builder.ask_for(&mut probe, Question::Const(extent), request.epoch, under);
            let Verdict::Constant(extent) = answer else {
                return Ok(Vec::new());
            };
            extents.push(extent.max(0));
        }
        // The capacities: the sum on the joined axis, CONST or MAX elsewhere.
        let boxes = builder.capacity_box(&mut probe, &array.dims, request.epoch, under);
        let mut out = Vec::new();
        for capacities in boxes {
            let mut capacities = capacities;
            capacities.axes[matched.axis] = extents.iter().sum();
            capacities.constant[matched.axis] = true;
            let mut candidate = Candidate::new(RuleId::Concat, Some("wire"), capacities.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched.push(key.clone());
            let mut filled = true;
            for (at, segment) in matched.segments.iter().enumerate() {
                // A rank-0 segment is one element: its port is the `Wire`
                // row the element is named by, and there is nothing to
                // enumerate. Every other segment is read at the grid the
                // output's box cuts out for it.
                let row = match (segment.rank, &segment.key) {
                    (0, _) => builder.child(
                        request,
                        segment.target.clone(),
                        Family::Wire,
                        request.condition,
                    )?,
                    (_, Some(key)) => {
                        let grid = self.segment_grid(&capacities, matched, extents[at]);
                        let batch = Batch::grid(&grid);
                        let key = key.clone();
                        builder.fetch_port(&mut candidate, request, &key, &batch)?
                    }
                    _ => {
                        filled = false;
                        break;
                    }
                };
                let Some(first) = row.entries.first() else {
                    filled = false;
                    break;
                };
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
            }
            if !filled {
                continue;
            }
            // `shape_j`: one wire per axis off the join whose extent is not
            // constant.
            for (axis, &constant) in capacities.constant.iter().enumerate() {
                if constant {
                    continue;
                }
                let row = builder.child(
                    request,
                    Target::Term(array.dims[axis]),
                    Family::Wire,
                    request.condition,
                )?;
                let Some(first) = row.entries.first() else {
                    filled = false;
                    break;
                };
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
            }
            if !filled {
                continue;
            }
            let slots = capacities.slots();
            candidate = candidate
                .output(
                    Target::Value(key.clone()),
                    Interface::Store(StoreIface {
                        owner: key.clone(),
                        dtype: array.dtype,
                        capacity: slots,
                        lambda: Layout::dense(
                            LayoutKind::Box(capacities.axes.clone()),
                            key.clone(),
                        ),
                    }),
                    request.condition,
                )
                .parameter("B", slots)
                .parameter("w", bits_of_capacity(slots) as i128);
            // A rank-1 box in C-order is a `Packed` on the same wires (§3.4).
            if capacities.axes.len() == 1 {
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

    /// The grid one segment's `Fetch` runs over: the output's capacities with
    /// the joined axis cut down to the segment's own extent, and dropped
    /// altogether for `E-Stack`, whose segments have one axis fewer.
    fn segment_grid(
        &self,
        capacities: &CapacityBox,
        matched: &Matched,
        extent: Capacity,
    ) -> Vec<Capacity> {
        let mut grid = capacities.axes.clone();
        if matched.stacked {
            grid.remove(matched.axis);
        } else {
            grid[matched.axis] = extent;
        }
        grid
    }

    /// `masked`: the segments' ordered bundles laid side by side, their
    /// validity bits concatenated. Rank 1, axis 0.
    fn in_place(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        matched: &Matched,
    ) -> Result<Vec<Candidate>, Reject> {
        if request.family != Family::Masked || array.rank() != 1 {
            return Ok(Vec::new());
        }
        let mut candidate = Candidate::new(RuleId::Concat, Some("masked"), request.epoch);
        candidate.matched.push(key.clone());
        let mut capacity: Capacity = 0;
        for segment in &matched.segments {
            // The in-place form lays whole bundles side by side, so it needs
            // a segment that is a value of Δ with a bundle of its own; a
            // rank-0 segment has one wire and is `wire`'s.
            let Some(segment) = segment.key.clone().filter(|_| segment.rank == 1) else {
                return Ok(Vec::new());
            };
            let mut found = None;
            for family in [Family::Store, Family::Masked, Family::Packed] {
                let psi = builder.psi(&segment);
                let condition =
                    builder.child_under_g(&mut candidate, request, psi, &segment.encode())?;
                let row =
                    builder.child(request, Target::Value(segment.clone()), family, condition)?;
                for entry in &row.entries {
                    let Some(lambda) = entry.interface.lambda() else {
                        continue;
                    };
                    if !lambda.layout.is_ordered() {
                        continue;
                    }
                    found = Some((entry.clone(), lambda.layout.slots()));
                    break;
                }
                if found.is_some() {
                    break;
                }
            }
            let Some((entry, slots)) = found else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(entry.requirement);
            candidate.epoch = candidate.epoch.max(entry.epoch);
            capacity += slots;
            // A `Packed` segment contributes the thermometer of its ℓ.
            if matches!(entry.interface, Interface::Packed { canon: false, .. }) {
                candidate = candidate.emit(Primitive::Therm { capacity: slots }, 1);
            }
        }
        // `len`: ℓ of the concatenation, a wire unless the extent is constant.
        if builder.terms.constant_value(array.dims[0]).is_none() {
            let row = builder.child(
                request,
                Target::Term(array.dims[0]),
                Family::Wire,
                request.condition,
            )?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        Ok(vec![candidate
            .output(
                Target::Value(key.clone()),
                Interface::Masked {
                    dtype: array.dtype,
                    capacity,
                    lambda: Layout::dense(
                        LayoutKind::Ranked { capacity, ranks: None },
                        key.clone(),
                    ),
                },
                request.condition,
            )
            .parameter("B", capacity)
            .parameter("w", bits_of_capacity(capacity) as i128)])
    }

    /// `pack`: the payload of `PACK_SEGMENTS`, whose contract already zeroes
    /// the tail, so the node adds no wire and no constraint.
    fn packed(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        matched: &Matched,
    ) -> Result<Vec<Candidate>, Reject> {
        if request.family != Family::Packed || array.rank() != 1 {
            return Ok(Vec::new());
        }
        let mut candidate = Candidate::new(RuleId::Concat, Some("pack"), request.epoch);
        candidate.matched.push(key.clone());
        let psi = builder.psi(&matched.list);
        let condition =
            builder.child_under_g(&mut candidate, request, psi, &matched.list.encode())?;
        let row = builder.child(
            request,
            Target::Value(matched.list.clone()),
            Family::Segments,
            condition,
        )?;
        let mut out = Vec::new();
        for entry in &row.entries {
            let Interface::Segments { capacity, .. } = &entry.interface else {
                continue;
            };
            let mut candidate = candidate.clone();
            candidate.inputs.push(entry.requirement);
            candidate.epoch = candidate.epoch.max(entry.epoch);
            let lambda = Layout::dense(LayoutKind::Compact(*capacity), key.clone());
            out.push(
                candidate
                    .output(
                        Target::Value(key.clone()),
                        Interface::Packed {
                            dtype: array.dtype,
                            capacity: *capacity,
                            lambda: lambda.clone(),
                            canon: true,
                        },
                        request.condition,
                    )
                    .output(
                        Target::Value(key.clone()),
                        Interface::Packed {
                            dtype: array.dtype,
                            capacity: *capacity,
                            lambda,
                            canon: false,
                        },
                        request.condition,
                    )
                    .parameter("B", *capacity),
            );
        }
        Ok(out)
    }
}
