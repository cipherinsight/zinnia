//! `GATHER` (`revision/phase1/rules/reading/gather.md`).
//!
//! Builds the read request of a value whose elements sit at positions named
//! by element values of an index array rather than by terms of Φ:
//! `take(a, I, axis=k)` and the one-advanced-entry subscript (E-Take), a
//! subscript with several advanced entries (E-Take-Broadcast), and `a[i]` or
//! `a[i_0, …]` with a scalar index that is not CONST (E-Index, E-Elem). For
//! each entry of the batch it fetches the index element of every advanced
//! entry, normalises a negative index, places the result on the gathered
//! axis of the source coordinate and applies the address of `README.md`
//! §3.2. Owner coordinates are wires, which a batch cannot name, so this
//! rule stands where `APPLY_INDEX_MAP` would.
//!
//! The in-range condition of the indices is phase 0's `β_I`: the rule assumes
//! it and `ENFORCE_CHECK` enforces it.
//!
//! **Which projection is offered.** The advanced entries are read off the
//! subscript, in source-axis order, and the block they broadcast to is the
//! one `gather.md` `source_pattern` prescribes: contiguous entries put it
//! where they sit, scattered ones at the front. Each entry is fetched at the
//! block coordinates of the output batch — the identity projection of
//! `gather.md` §3.3, which an index array whose extents are the block's own
//! gives. An entry whose extents differ from the block's would need the
//! `ite` coordinate of the broadcast projection; it is not offered, and the
//! row says so.

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::ast_index::SubscriptEntry;
use crate::phase1::build::Builder;
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, BatchEntry, Capacity, Coordinate, Family, Interface, RangeAxis,
};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::{RowEntry, RowRequest};
use crate::phase1::Reject;

pub(crate) struct Gather;

/// One advanced entry of the subscript (`gather.md`, `source_pattern`): the
/// index value, the source axis it addresses and its rank.
#[derive(Debug, Clone)]
pub(crate) struct Advanced {
    pub key: ValueKey,
    pub axis: usize,
    /// The rank of the index array; 0 for a scalar entry.
    pub rank: usize,
}

/// What the source pattern read: the source array and its advanced entries.
pub(crate) struct Match {
    pub source: ValueKey,
    /// The first advanced entry, which is the whole match of a one-entry
    /// take; `term_to_wires.md`'s `β_I` row reads it.
    pub index: ValueKey,
    pub axis: usize,
    /// The rank of the broadcast index block.
    pub block: usize,
    /// Every advanced entry, in source-axis order.
    pub entries: Vec<Advanced>,
    /// The output axis at which the block sits (`source_pattern`): where the
    /// advanced entries sit when they are adjacent, and 0 otherwise.
    pub block_start: usize,
}

/// The take a value is, for a rule outside this one: `TERM_TO_WIRES` needs it
/// to compile `β_I`, the in-range condition of the same index array
/// (`term_to_wires.md`, the `β_I` row).
pub(crate) fn take_of(builder: &Builder<'_>, value: &ValueKey) -> Option<Match> {
    Gather.matches(builder, value)
}

impl Rule for Gather {
    fn id(&self) -> RuleId {
        RuleId::Gather
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::ReadReq]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let (Some(value), Some(batch)) = (request.target.as_value(), request.batch.clone()) else {
            return Ok(Vec::new());
        };
        let value = value.clone();
        let Some(matched) = self.matches(builder, &value) else {
            return Ok(Vec::new());
        };
        let Some(source_dims) = builder.array(&matched.source).map(|array| array.dims.clone())
        else {
            return Ok(Vec::new());
        };

        let mut probe = Candidate::new(RuleId::Gather, None, request.epoch);
        // `coords`: the output-side coordinates, as for every read request.
        let coords = builder.child_in_batch(
            request,
            request.target.clone(),
            Family::Coords,
            request.condition,
        )?;
        let Some(coords) = coords.entries.first().cloned() else {
            return Ok(Vec::new());
        };
        // `idx(m)`: the index elements of every advanced entry, each at the
        // block coordinates of the batch, with `Q_m` and `w_I(m)` of the
        // emitter's constraint count.
        let mut fetched: Vec<(RowEntry, Capacity, u32)> =
            Vec::with_capacity(matched.entries.len());
        for advanced in &matched.entries {
            let Some(index_batch) =
                self.index_batch(builder, &mut probe, request, &value, &batch, &matched, advanced)
            else {
                return Ok(Vec::new());
            };
            let queries = index_batch.len();
            let row = builder.fetch_port(&mut probe, request, &advanced.key, &index_batch)?;
            let Some(first) = row.entries.first().cloned() else {
                return Ok(Vec::new());
            };
            let width = builder.omega.round(builder.dtype_width(self.index_dtype(builder, &advanced.key)));
            fetched.push((first, queries, width));
        }
        // `ext(m)`: the extent of each gathered axis, when it is not
        // constant. The port is the same for every owner, so it is raised
        // once and a missing one drops the rule.
        let under = builder.under(request.condition);
        let mut extents: Vec<RowEntry> = Vec::new();
        let mut asked: Vec<TermId> = Vec::new();
        for advanced in &matched.entries {
            let extent = source_dims[advanced.axis];
            if asked.contains(&extent) {
                continue;
            }
            asked.push(extent);
            let answer = builder.ask_for(
                &mut probe,
                Question::Const(extent),
                request.epoch,
                under,
            );
            if matches!(answer, Verdict::Constant(_)) {
                continue;
            }
            let row =
                builder.child(request, Target::Term(extent), Family::Wire, request.condition)?;
            let Some(first) = row.entries.first().cloned() else {
                return Ok(Vec::new());
            };
            extents.push(first);
        }

        let mut out = Vec::new();
        // `base`: one node per Store-family provider of the source. A `Ref`
        // owner is not offered here: composing a gather with a view is
        // `reading/README.md` Open 1.
        for family in [Family::Store, Family::Masked, Family::Packed] {
            let psi = builder.psi(&matched.source);
            let condition =
                builder.child_under_g(&mut probe, request, psi, &matched.source.encode())?;
            let row = builder.child(
                request,
                Target::Value(matched.source.clone()),
                family,
                condition,
            )?;
            for provider in &row.entries {
                let Some(owner) = super::owner_of(&provider.interface, &matched.source) else {
                    continue;
                };
                if !owner.is_dense() && source_dims.len() > 1 {
                    continue;
                }
                if super::identity_strides(builder, &owner, &source_dims).is_none() {
                    continue;
                }
                let mut candidate = Candidate::new(RuleId::Gather, None, request.epoch);
                candidate.answers = probe.answers.clone();
                candidate.matched = vec![value.clone(), matched.source.clone()];
                candidate.matched.extend(
                    matched.entries.iter().map(|advanced| advanced.key.clone()),
                );
                candidate.inputs.push(coords.requirement);
                candidate.epoch = candidate.epoch.max(coords.epoch);
                candidate.inputs.push(provider.requirement);
                candidate.epoch = candidate.epoch.max(provider.epoch);
                for (index, _, _) in &fetched {
                    candidate.inputs.push(index.requirement);
                    candidate.epoch = candidate.epoch.max(index.epoch);
                }
                for extent in &extents {
                    candidate.inputs.push(extent.requirement);
                    candidate.epoch = candidate.epoch.max(extent.epoch);
                }
                // emitter: per fetched index the sign test and the select of
                // the normalisation; the address is then a `lin`.
                for &(_, queries, width) in &fetched {
                    candidate = candidate
                        .emit(Primitive::Cmp { width }, queries)
                        .emit(Primitive::Select, queries);
                }
                candidate = candidate
                    .output(
                        request.target.clone(),
                        Interface::ReadReq {
                            owner: owner.clone(),
                            queries: batch.len(),
                            // A gathered address is a wire: its coordinate on
                            // the gathered axis is an element of I.
                            constant: false,
                        },
                        request.condition,
                    )
                    .parameter("Q", batch.len())
                    .parameter("B", owner.slots());
                out.push(candidate);
            }
        }
        Ok(out)
    }
}

impl Gather {
    /// `source_pattern`: the producer names E-Take, E-Take-Broadcast, E-Index
    /// or E-Elem, and the operand list holds the source array and the index
    /// value of every advanced entry.
    fn matches(&self, builder: &Builder<'_>, value: &ValueKey) -> Option<Match> {
        let entry = builder.entry(value)?;
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return None;
        };
        if !rules.iter().any(|rule| {
            matches!(
                rule,
                Phase0Rule::ETake
                    | Phase0Rule::ETakeBroadcast
                    | Phase0Rule::EIndex
                    | Phase0Rule::EElem
            )
        }) {
            return None;
        }
        let source = operands
            .iter()
            .find(|operand| builder.array(operand).is_some())?
            .clone();
        let source_rank = builder.array(&source).map(|array| array.rank()).unwrap_or(1);
        let entries = self
            .subscript_entries(builder, value, &source, source_rank)
            .or_else(|| self.take_entries(builder, value, operands, &source, source_rank))?;
        let first = entries.first()?.clone();
        let block = entries.iter().map(|advanced| advanced.rank).max().unwrap_or(0);
        let output_rank = builder.array(value).map(|array| array.rank()).unwrap_or(0);
        // The block is a set of w's own axes, so w has at least that rank
        // ("ranks as E-Take-Broadcast prescribes"). A rank-0 w under a
        // rank-≥1 index block is a site whose value is not a gather at all:
        // phase 0 records `ravel_multi_index` as an E-Take over the index
        // array and the shape array, and `reading/README.md` §4 leaves it to
        // `elementwise`.
        if output_rank < block {
            return None;
        }
        Some(Match {
            source,
            index: first.key,
            axis: first.axis,
            block,
            block_start: Self::block_start(&entries, source_rank, block, output_rank),
            entries,
        })
    }

    /// The advanced entries of a subscript, in source-axis order, read off
    /// the AST node and Δ: an entry whose value is an integer array, or a
    /// scalar that is the index of an `E-Index` or `E-Elem`.
    fn subscript_entries(
        &self,
        builder: &Builder<'_>,
        value: &ValueKey,
        source: &ValueKey,
        source_rank: usize,
    ) -> Option<Vec<Advanced>> {
        let node = builder.node(value)?;
        let written = node.entries()?;
        // The Ellipsis stands for the `:` entries the subscript omits, which
        // consume source axes and are never advanced.
        let consuming = written
            .iter()
            .filter(|entry| !matches!(entry, SubscriptEntry::NewAxis | SubscriptEntry::Ellipsis))
            .count();
        let mut out = Vec::new();
        let mut axis = 0usize;
        for entry in written {
            match entry {
                SubscriptEntry::NewAxis => {}
                SubscriptEntry::Ellipsis => axis += source_rank.saturating_sub(consuming),
                SubscriptEntry::Slice { .. } => axis += 1,
                SubscriptEntry::Single { expr_index } => {
                    let sibling = builder.sibling(value, *expr_index)?;
                    if sibling.key == *source {
                        return None;
                    }
                    let rank = match builder.array(&sibling.key) {
                        Some(array) => array.rank(),
                        None => {
                            builder.scalar(&sibling.key)?;
                            0
                        }
                    };
                    out.push(Advanced { key: sibling.key.clone(), axis, rank });
                    axis += 1;
                }
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// `np.take(a, I, axis=k)` and every other site with no subscript node:
    /// the one index operand beside the source, on the axis the call names.
    fn take_entries(
        &self,
        builder: &Builder<'_>,
        value: &ValueKey,
        operands: &[ValueKey],
        source: &ValueKey,
        source_rank: usize,
    ) -> Option<Vec<Advanced>> {
        let index = operands
            .iter()
            .filter(|operand| *operand != source)
            .find(|operand| {
                builder
                    .array(operand)
                    .map(|array| matches!(array.dtype, Dtype::I64 | Dtype::Idx))
                    .unwrap_or(false)
            })
            .or_else(|| {
                operands
                    .iter()
                    .find(|operand| builder.scalar(operand).is_some())
            })?
            .clone();
        let rank = builder.array(&index).map(|array| array.rank()).unwrap_or(0);
        let axis = builder
            .node(value)
            .and_then(|node| crate::phase1::rules::parameter(node, 2, "axis"))
            .and_then(|value| value.int())
            .map(|value| if value < 0 { value + source_rank as i64 } else { value })
            .unwrap_or(0)
            .max(0) as usize;
        Some(vec![Advanced { key: index, axis, rank }])
    }

    /// The output axis at which the index block sits (`source_pattern`):
    /// where the advanced entries sit when they are adjacent source axes,
    /// and the front otherwise.
    ///
    /// The axes of w are the basic source axes in order with the block
    /// inserted, so their count adds up to r_w when the site has that shape.
    /// Where it does not, the block is the front: `take_along_axis`, which
    /// phase 0 also records as E-Take, gives w the index's own shape, and
    /// then the whole of w is the block.
    fn block_start(
        entries: &[Advanced],
        source_rank: usize,
        block: usize,
        output_rank: usize,
    ) -> usize {
        let adjacent = entries
            .windows(2)
            .all(|pair| pair[1].axis == pair[0].axis + 1);
        let basic = source_rank.saturating_sub(entries.len());
        let first = entries.first().map(|advanced| advanced.axis).unwrap_or(0);
        if adjacent && basic + block == output_rank && first <= basic {
            first
        } else {
            0
        }
    }

    /// `batch_m`: the block coordinates at which the index elements of one
    /// advanced entry are read (`gather.md`, `idx(m)`).
    ///
    /// The block occupies `block` axes of the output from `block_start`, and
    /// an index array of rank `r ≤ block` is right-aligned in it. On a block
    /// axis whose extent the entry has, the projection is the identity and
    /// the batch keeps the output's own coordinate there; on one whose
    /// extent CONST answers 1, the entry is broadcast and every output
    /// coordinate reads position 0, which a range batch writes as one
    /// template of width 1 — the distinct projections of `idx(m)`. An axis
    /// of neither kind needs the `ite` coordinate and is not offered
    /// (`gather.md` Open 4).
    fn index_batch(
        &self,
        builder: &mut Builder<'_>,
        probe: &mut Candidate,
        request: &RowRequest,
        value: &ValueKey,
        batch: &Batch,
        matched: &Match,
        advanced: &Advanced,
    ) -> Option<Batch> {
        if advanced.rank == 0 {
            return Some(Batch::Scalar);
        }
        let from = matched.block_start + (matched.block - advanced.rank);
        if from + advanced.rank > batch.rank() {
            return None;
        }
        // Which block axes the entry is broadcast along, one CONST question
        // per axis whose extent is not the block's.
        let index_dims = builder.array(&advanced.key)?.dims.clone();
        let output_dims = builder.array(value)?.dims.clone();
        let under = builder.under(request.condition);
        let mut broadcast = Vec::with_capacity(advanced.rank);
        for at in 0..advanced.rank {
            if index_dims.get(at)? == output_dims.get(from + at)? {
                broadcast.push(false);
                continue;
            }
            let answer = builder.ask_for(
                probe,
                Question::Const(*index_dims.get(at)?),
                request.epoch,
                under,
            );
            if !matches!(answer, Verdict::Constant(1)) {
                return None;
            }
            broadcast.push(true);
        }
        match batch {
            Batch::Range { axes: templates, live } => Some(Batch::Range {
                axes: (0..advanced.rank)
                    .map(|at| match broadcast[at] {
                        false => templates[from + at].clone(),
                        true => RangeAxis {
                            base: Coordinate::Literal(0),
                            stride: 1,
                            width: 1,
                            broadcast: None,
                        },
                    })
                    .collect(),
                live: *live,
            }),
            Batch::General(entries) => Some(Batch::General(
                entries
                    .iter()
                    .map(|entry| BatchEntry {
                        coordinate: (0..advanced.rank)
                            .map(|at| match broadcast[at] {
                                false => entry.coordinate[from + at].clone(),
                                true => Coordinate::Literal(0),
                            })
                            .collect(),
                        live: entry.live,
                    })
                    .collect(),
            )),
            Batch::Scalar => None,
        }
    }

    /// `w_I(m)`, the width of the index elements of one advanced entry.
    fn index_dtype(&self, builder: &Builder<'_>, index: &ValueKey) -> Dtype {
        builder
            .array(index)
            .map(|array| array.dtype)
            .or_else(|| builder.scalar(index).map(|scalar| scalar.dtype))
            .unwrap_or(Dtype::I64)
    }
}
