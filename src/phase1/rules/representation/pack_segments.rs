//! `PACK_SEGMENTS` (`revision/phase1/rules/representation/pack_segments.md`).
//!
//! Places n one-dimensional values end to end in one payload owned by the
//! node: segment k occupies `[off_k, off_k + ℓ_k)` with `off_1 = 0` and
//! `off_{k+1} = off_k + ℓ_k`, and every position beyond `Σ_k ℓ_k` holds 0.
//! The node is attached to `n + 1` requirements (§3.4): the `Segments`
//! bundle `CONCAT` consumes, and one `Ref` per segment through which a
//! reader addresses the payload at a run-time offset, so that n segments
//! share one ROM context.
//!
//! **Which value is matched.** The Composite `L = [v_1, …, v_n]` the
//! `concatenate` call took as its argument, which Δ keys as one value whose
//! producer is an `E-Composite` with one operand per element; the operands
//! are the segments ([`crate::phase1::build::Builder::element_target`]).
//!
//! **How many nodes.** One per combination of the concrete `Packed` bundles
//! the segments' rows offer, which is what `required_input_ports` asks for;
//! `Ω.node_budget` is the only bound on the product, and the corpus's
//! concatenations have two or three segments with one or two bundles each.

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::term::TermId;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::{ReqId, Target};
use crate::phase1::interface::{
    Capacity, Extent, Family, Interface, LayoutKind, Mu, MuForm, OwnerRef, RefIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct PackSegments;

/// One segment as the ports resolved it.
#[derive(Clone)]
struct Segment {
    value: ValueKey,
    requirement: ReqId,
    epoch: u32,
    /// `B_k`, the capacity of the segment's own payload.
    capacity: Capacity,
    /// Whether the segment's tail is already zero (`canon = yes`).
    canon: bool,
    /// `dims_{v_k}[0]`, the segment's length term.
    length: TermId,
}

impl Rule for PackSegments {
    fn id(&self) -> RuleId {
        RuleId::PackSegments
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Segments, Family::Ref]
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
        let Some(elements) = self.matches(builder, &key) else {
            return Ok(Vec::new());
        };
        let Some(dtype) = self.dtype(builder, &elements) else {
            return Ok(Vec::new());
        };

        // premises and resource_relations: `B = MAX Σ_k dims_{v_k}[0]` over
        // Φ_e ∧ g, clamped by `Σ_k B_k`; `CONST Σ_{j<k} dims_{v_j}[0]` per
        // segment decides whether it is wired or shifted.
        let mut probe = Candidate::new(RuleId::PackSegments, None, request.epoch);
        let lengths: Vec<TermId> = elements
            .iter()
            .filter_map(|value| builder.array(value).map(|array| array.dims[0]))
            .collect();
        if lengths.len() != elements.len() {
            return Ok(Vec::new());
        }
        let total = self.sum(builder, &lengths);
        let under = builder.under(request.condition);
        let boxes = builder.capacity_box(&mut probe, &[total], request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);

        // The `Packed` bundles of every segment, one node per combination.
        let mut rows: Vec<Vec<Segment>> = Vec::with_capacity(elements.len());
        for (at, value) in elements.iter().enumerate() {
            let psi = builder.psi(value);
            let condition = builder.child_under_g(&mut probe, request, psi, &value.encode())?;
            let row = builder.child(request, Target::Value(value.clone()), Family::Packed, condition)?;
            let mut offered = Vec::new();
            for entry in &row.entries {
                let Interface::Packed { capacity, canon, .. } = &entry.interface else {
                    continue;
                };
                offered.push(Segment {
                    value: value.clone(),
                    requirement: entry.requirement,
                    epoch: entry.epoch,
                    capacity: *capacity,
                    canon: *canon,
                    length: lengths[at],
                });
            }
            if offered.is_empty() {
                return Ok(Vec::new());
            }
            rows.push(offered);
        }

        let mut out = Vec::new();
        for combination in combinations(&rows) {
            for capacities in &boxes {
                let held: Capacity = combination.iter().map(|segment| segment.capacity).sum();
                let capacity = capacities.axes[0].min(held);
                let mut candidate =
                    Candidate::new(RuleId::PackSegments, None, capacities.epoch);
                candidate.answers = probe.answers.clone();
                candidate.matched.push(key.clone());
                candidate.epoch = candidate.epoch.max(builder.watermark(&key));
                for segment in &combination {
                    candidate.inputs.push(segment.requirement);
                    candidate.epoch = candidate.epoch.max(segment.epoch);
                }
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

                // The offsets, and the emitter that places each segment.
                let width = bits_of_capacity(capacity);
                let mut offsets: Vec<Option<Capacity>> = Vec::with_capacity(combination.len());
                let mut prefix: Vec<TermId> = Vec::new();
                for segment in &combination {
                    let offset = self.sum(builder, &prefix);
                    let answer = builder.ask_for(
                        &mut candidate,
                        Question::Const(offset),
                        capacities.epoch,
                        under,
                    );
                    offsets.push(match answer {
                        Verdict::Constant(value) => Some(value),
                        _ => None,
                    });
                    prefix.push(segment.length);
                }
                for (at, segment) in combination.iter().enumerate() {
                    // Tail masking, omitted where the segment is canonical or
                    // its length fills its capacity.
                    let full = builder.terms.constant_value(segment.length)
                        == Some(segment.capacity);
                    if !segment.canon && !full {
                        candidate = candidate
                            .emit(Primitive::Therm { capacity: segment.capacity }, 1)
                            .emit(Primitive::Mul, segment.capacity);
                    }
                    // A run-time offset is a barrel shifter of `bits(B)`
                    // stages over the payload.
                    if offsets[at].is_none() {
                        candidate = candidate
                            .emit(Primitive::Range { width }, 1)
                            .emit(Primitive::Select, width as i128 * capacity);
                    }
                }
                candidate = candidate
                    .emit(Primitive::Mul, i128::from(gated) * combination.len() as i128)
                    .output(
                        Target::Value(key.clone()),
                        Interface::Segments {
                            owner: key.clone(),
                            dtype,
                            capacity,
                            segments: combination.len(),
                        },
                        request.condition,
                    )
                    .parameter("B", capacity)
                    .parameter("n", combination.len() as i128)
                    .parameter("w", width as i128);
                // One `Ref` per segment, through which a reader addresses the
                // payload at the segment's offset (§3.4).
                let owner = OwnerRef {
                    value: key.clone(),
                    family: Family::Segments,
                    capacity,
                    layout: LayoutKind::Compact(capacity),
                    canon: false,
                };
                for (at, segment) in combination.iter().enumerate() {
                    let (map, params) = match offsets[at] {
                        Some(0) => (Mu::Coord(0), Vec::new()),
                        Some(offset) => (
                            Mu::Add(Box::new(Mu::Const(offset)), Box::new(Mu::Coord(0))),
                            Vec::new(),
                        ),
                        None => (
                            Mu::Add(Box::new(Mu::Param(0)), Box::new(Mu::Coord(0))),
                            vec![self.sum(
                                builder,
                                &combination[..at]
                                    .iter()
                                    .map(|segment| segment.length)
                                    .collect::<Vec<_>>(),
                            )],
                        ),
                    };
                    let extent = match builder.terms.constant_value(segment.length) {
                        Some(value) => Extent::Literal(value),
                        None => Extent::Term(segment.length),
                    };
                    candidate = candidate.output(
                        Target::Value(segment.value.clone()),
                        Interface::Ref(RefIface {
                            owner: owner.clone(),
                            form: MuForm::Key,
                            map: vec![map],
                            params,
                            ext: vec![extent],
                            span: 1,
                            nested: None,
                        }),
                        builder.psi(&segment.value),
                    );
                }
                out.push(candidate);
            }
        }
        Ok(out)
    }
}

impl PackSegments {
    /// `source_pattern`: a Composite of n ≥ 1 rank-1 array elements, each of
    /// which Δ names — the `E-Composite` operand list.
    fn matches(&self, builder: &Builder<'_>, key: &ValueKey) -> Option<Vec<ValueKey>> {
        let composite = builder.entry(key)?.desc.as_composite()?;
        if composite.elems.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(composite.elems.len());
        for at in 0..composite.elems.len() {
            let array = composite.elems[at].as_array()?;
            if array.rank() != 1 {
                return None;
            }
            match builder.element_target(key, &[at])? {
                Target::Value(value) => out.push(value),
                _ => return None,
            }
        }
        Some(out)
    }

    /// The one dtype of the segments (`premises`, structural).
    fn dtype(&self, builder: &Builder<'_>, elements: &[ValueKey]) -> Option<Dtype> {
        let mut dtype = None;
        for value in elements {
            let found = builder.array(value)?.dtype;
            if !super::super::sources::has_encoding(found) {
                return None;
            }
            if *dtype.get_or_insert(found) != found {
                return None;
            }
        }
        dtype
    }

    /// `Σ terms`, built through the arena.
    fn sum(&self, builder: &mut Builder<'_>, terms: &[TermId]) -> TermId {
        let mut total = builder.literal(0);
        for &term in terms {
            total = builder.terms.add(total, term);
        }
        total
    }
}

/// Every choice of one entry per row, in row order.
fn combinations(rows: &[Vec<Segment>]) -> Vec<Vec<Segment>> {
    let mut out: Vec<Vec<Segment>> = vec![Vec::new()];
    for row in rows {
        let mut next = Vec::with_capacity(out.len() * row.len());
        for prefix in &out {
            for entry in row {
                let mut extended = prefix.clone();
                extended.push(entry.clone());
                next.push(extended);
            }
        }
        out = next;
    }
    out
}
