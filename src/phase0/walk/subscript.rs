//! Subscripts: the indexing and slicing rules of `definition.md` §5.4 and the
//! boolean indexing of §5.5, for reads and for `S-SetItem` (§5.11).
//!
//! A subscript is a list of per-axis entries, which the front end exports as
//! `ASTSlice.data`: an expression node, a `[lo, hi, step]` triple, or one of
//! the two sentinels for `...` and `np.newaxis`
//! (`zinnia/compile/transformer/base.py`'s `visit_slice_key`). The entries
//! are applied left to right, each to the descriptor the previous one
//! produced, so every form is one rule of §5.4 or §5.5 and the composition is
//! the only thing this module adds.

use serde::Deserialize;

use crate::ast::{ASTNode, ASTSlice, ASTSubscriptAssignTarget};
use crate::phase0::desc::{ArrayDesc, CompositeDesc, Desc, Dtype, ScalarDesc, ValueKey};
use crate::phase0::rules::helpers::in_bounds;
use crate::phase0::rules::require_int_term;
use crate::phase0::rules::{
    broadcast_dims, e_composite, e_composite_index, e_composite_slice, e_elem, e_expand_dims,
    e_filter_axis0, e_filter_same, e_index, e_slice, e_take, merge_desc, slice_extent,
};
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;
use crate::phase0::Reject;

use super::Walker;
use crate::phase0::rules::Rule;

/// One entry of a subscript.
pub(crate) enum Index {
    /// An expression entry, with the key of the value it evaluated to: a
    /// Boolean mask is what `Count` is keyed by (§5.5).
    Single {
        desc: Desc,
        key: ValueKey,
    },
    Range {
        lo: Option<Desc>,
        hi: Option<Desc>,
        step: Option<Desc>,
    },
    Ellipsis,
    NewAxis,
}

impl Index {
    /// How many axes of the indexed array the entry consumes.
    fn axes(&self) -> usize {
        match self {
            Index::Single { desc, .. } => match desc.as_array() {
                // A boolean mask consumes as many axes as it has.
                Some(a) if a.dtype == Dtype::Bool => a.rank(),
                Some(_) => 1,
                None => 1,
            },
            Index::Range { .. } => 1,
            Index::Ellipsis | Index::NewAxis => 0,
        }
    }
}

/// One entry of an `ASTSlice` as the front end exports it.
///
/// This is the whole shape of that part of the schema, in one place: the two
/// marker nodes, a `[lo, hi, step]` triple with `null` for an omitted
/// endpoint, and anything else as an ordinary expression node. Reading it
/// through serde is what keeps the two string literals out of the walker.
#[derive(Deserialize)]
#[serde(untagged)]
enum SliceEntry {
    Ellipsis(EllipsisMarker),
    NewAxis(NewAxisMarker),
    /// `lo:hi:step`, each endpoint present, `null` or absent.
    Range(Vec<Option<ASTNode>>),
    Single(ASTNode),
}

#[derive(Deserialize)]
#[serde(tag = "__class__")]
enum EllipsisMarker {
    ASTSliceEllipsis {},
}

#[derive(Deserialize)]
#[serde(tag = "__class__")]
enum NewAxisMarker {
    ASTSliceNewAxis {},
}

/// One subscript entry after its operand has been coerced to what it denotes:
/// the four forms §5.4 and §5.5 give a rule to, plus `np.newaxis`.
enum Entry<'i> {
    /// An integer scalar: `E-Index` (§5.4), or `E-Elem` on a rank-1 array.
    Scalar(ScalarDesc),
    /// A Boolean array: `E-Filter` (§5.5), with the key of the mask value
    /// its count is keyed by.
    Mask {
        desc: ArrayDesc,
        key: ValueKey,
    },
    /// An integer array: `E-Take` (§5.4).
    Advanced(ArrayDesc),
    /// `lo:hi:step`: `E-Slice` (§5.4). The bounds stay on the node, because
    /// they are already-evaluated descriptors there.
    Slice(&'i Index),
    NewAxis,
}

/// One output axis of a multi-index subscript: an extent the entries left
/// alone, or the place the broadcast block of the advanced indices goes.
enum Slot {
    Kept(TermId),
    Block,
}

/// Put the broadcast block into the slots, following NumPy's rule: in place
/// when the advanced indices are adjacent, at the front when a slice or a new
/// axis separates them.
fn place_block(slots: &[Slot], block: &[TermId]) -> Vec<TermId> {
    let positions: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| matches!(slot, Slot::Block))
        .map(|(at, _)| at)
        .collect();
    let adjacent = positions.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let insert_at = if adjacent {
        positions.first().copied()
    } else {
        None
    };
    let mut dims = Vec::with_capacity(slots.len() + block.len());
    if insert_at.is_none() {
        dims.extend_from_slice(block);
    }
    for (at, slot) in slots.iter().enumerate() {
        match slot {
            Slot::Kept(d) => dims.push(*d),
            Slot::Block if Some(at) == insert_at => dims.extend_from_slice(block),
            Slot::Block => {}
        }
    }
    dims
}

impl Walker<'_> {
    /// Evaluate the entries of a subscript, left to right.
    ///
    /// An omitted slice endpoint arrives as an `ASTConstantNone` node, which
    /// the baseline visits like any other expression, so it is evaluated here
    /// too and its `Desc::None` result means "omitted".
    pub fn slice_indices(&mut self, slicing: &ASTSlice) -> Result<Vec<Index>, Reject> {
        let mut indices = Vec::with_capacity(slicing.data.len());
        for entry in &slicing.data {
            let parsed: SliceEntry =
                serde_json::from_value(entry.clone()).map_err(|e| Reject::Unsupported {
                    construct: format!("unreadable subscript entry: {}", e),
                })?;
            indices.push(match parsed {
                SliceEntry::Ellipsis(_) => Index::Ellipsis,
                SliceEntry::NewAxis(_) => Index::NewAxis,
                SliceEntry::Range(triple) => Index::Range {
                    lo: self.slice_endpoint(triple.first())?,
                    hi: self.slice_endpoint(triple.get(1))?,
                    step: self.slice_endpoint(triple.get(2))?,
                },
                SliceEntry::Single(node) => {
                    let (desc, key) = self.expr_keyed(&node)?;
                    Index::Single { desc, key }
                }
            });
        }
        Ok(indices)
    }

    fn slice_endpoint(&mut self, entry: Option<&Option<ASTNode>>) -> Result<Option<Desc>, Reject> {
        let Some(Some(node)) = entry else {
            return Ok(None);
        };
        match self.expr(node)? {
            Desc::None => Ok(None),
            other => Ok(Some(other)),
        }
    }

    /// `v[...]`: the read form.
    pub fn subscript(
        &mut self,
        value: &Desc,
        indices: &[Index],
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        match value {
            Desc::Composite(c) => self.subscript_composite(c, indices, key),
            Desc::Array(a) => {
                let a = a.clone();
                self.subscript_array(&a, indices, key)
            }
            other => Err(Reject::Unsupported {
                construct: format!("subscript of a {}", other.kind_str()),
            }),
        }
    }

    /// `E-Composite-Index` and the composite slice of §5.11. A composite
    /// indexed by anything but one scalar or one constant-bounded slice is
    /// treated as the array it denotes, which is what the baseline does after
    /// materialising it.
    fn subscript_composite(
        &mut self,
        c: &CompositeDesc,
        indices: &[Index],
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        if let [Index::Single { desc, .. }] = indices {
            if let Some(scalar) = desc.as_scalar() {
                let (c, scalar) = (c.clone(), scalar.clone());
                return e_composite_index(&mut self.ctx(), &c, &scalar, key, "item");
            }
        }
        if let [Index::Range { lo, hi, step }] = indices {
            let step = self.constant_step(step.as_ref())?;
            let (lo, hi) = (as_scalar(lo.as_ref()), as_scalar(hi.as_ref()));
            let c = c.clone();
            return e_composite_slice(&mut self.ctx(), &c, lo.as_ref(), hi.as_ref(), step);
        }
        if let [Index::Ellipsis] = indices {
            return Ok(Desc::Composite(c.clone()));
        }
        let array = self.as_array(&Desc::Composite(c.clone()), Rule::EIndex)?;
        self.subscript_array(&array, indices, key)
    }

    /// The array forms of §5.4 and §5.5.
    ///
    /// One integer-array index is `E-Take` applied to one axis, and a chain
    /// of independent entries composes left to right, which is how §5.4
    /// states each rule. Two or more of them do *not* compose: NumPy
    /// broadcasts every advanced index against the others and produces one
    /// block of the broadcast shape, so that case goes to
    /// [`Walker::subscript_broadcast`].
    pub fn subscript_array(
        &mut self,
        a: &ArrayDesc,
        indices: &[Index],
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let expanded = expand_ellipsis(indices, a.rank())?;
        let mut entries = Vec::with_capacity(expanded.len());
        for index in &expanded {
            entries.push(self.entry(index)?);
        }
        let advanced = entries
            .iter()
            .filter(|e| matches!(e, Entry::Scalar(_) | Entry::Advanced(_)))
            .count();
        let arrays = entries
            .iter()
            .filter(|e| matches!(e, Entry::Advanced(_)))
            .count();
        if arrays > 0 && advanced > 1 {
            return self.subscript_broadcast(a, &entries, key);
        }
        let mut current = Desc::Array(a.clone());
        let mut axis = 0usize;
        for (position, entry) in entries.iter().enumerate() {
            let array = match &current {
                Desc::Array(array) => array.clone(),
                other => {
                    return Err(Reject::Unsupported {
                        construct: format!(
                            "subscript entry {} applied to a {}",
                            position,
                            other.kind_str()
                        ),
                    })
                }
            };
            current = self.apply_entry(&array, entry, &mut axis, key, position)?;
        }
        Ok(current)
    }

    /// One subscript entry with its operand coerced to what it denotes.
    ///
    /// The coercion happens once, before the entries are classified: turning
    /// a composite index into an array is itself a rule (`E-Stack`, §5.6) and
    /// applying it twice would record its check twice.
    fn entry<'i>(&mut self, index: &'i Index) -> Result<Entry<'i>, Reject> {
        match index {
            Index::Ellipsis => unreachable!("expand_ellipsis removed every ellipsis"),
            Index::NewAxis => Ok(Entry::NewAxis),
            Index::Range { .. } => Ok(Entry::Slice(index)),
            Index::Single {
                desc: Desc::Scalar(s),
                ..
            } => Ok(Entry::Scalar(s.clone())),
            Index::Single { desc, key } => {
                let indexer = self.as_array(desc, Rule::ETake)?;
                Ok(if indexer.dtype == Dtype::Bool {
                    // `Count` is keyed by the mask value (§5.5), so the entry
                    // carries the value the index names rather than the index
                    // itself: two reads of one mask then share one count.
                    Entry::Mask {
                        desc: indexer,
                        key: self.an.defining_key(key),
                    }
                } else {
                    Entry::Advanced(indexer)
                })
            }
        }
    }

    /// One subscript entry (§5.4, §5.5). `axis` is the axis of the *current*
    /// descriptor the entry applies to; an entry that removes an axis leaves
    /// it, an entry that keeps one advances it.
    fn apply_entry(
        &mut self,
        a: &ArrayDesc,
        entry: &Entry,
        axis: &mut usize,
        key: &ValueKey,
        position: usize,
    ) -> Result<Desc, Reject> {
        match entry {
            Entry::NewAxis => {
                let inserted = e_expand_dims(&mut self.ctx(), a, *axis as i64)?;
                *axis += 1;
                Ok(inserted)
            }
            Entry::Slice(index) => {
                let (lo, hi, step) = self.slice_parts(index)?;
                let sliced = e_slice(&mut self.ctx(), a, *axis, lo.as_ref(), hi.as_ref(), step)?;
                *axis += 1;
                Ok(sliced)
            }
            Entry::Scalar(s) => {
                let s = s.clone();
                self.index_by_scalar(a, &s, *axis, key, position)
            }
            Entry::Mask { desc, key } => {
                let (mask, mask_key) = (desc.clone(), key.clone());
                self.index_by_mask(a, &mask, &mask_key, axis, position)
            }
            Entry::Advanced(indexer) => {
                let indexer = indexer.clone();
                let base = format!("c{}", position);
                let taken = e_take(&mut self.ctx(), a, &indexer, *axis, key, &base)?;
                *axis += indexer.rank();
                Ok(taken)
            }
        }
    }

    /// The evaluated bounds of a slice entry, with its constant step (§5.4).
    #[allow(clippy::type_complexity)]
    fn slice_parts(
        &mut self,
        index: &Index,
    ) -> Result<(Option<ScalarDesc>, Option<ScalarDesc>, i64), Reject> {
        let Index::Range { lo, hi, step } = index else {
            return Err(Reject::Unsupported {
                construct: "a slice entry that is not a slice".to_string(),
            });
        };
        let step = self.constant_step(step.as_ref())?;
        Ok((as_scalar(lo.as_ref()), as_scalar(hi.as_ref()), step))
    }

    /// Several advanced indices in one subscript: NumPy's broadcasting rule
    /// (`E-Take`, §5.4, generalised).
    ///
    /// Every advanced index — an integer array, and an integer scalar, which
    /// is a rank-0 array — is broadcast against the others into one block of
    /// dimensions. The block replaces the axes those entries consumed, in
    /// place when they are adjacent and at the front when a slice separates
    /// them; the remaining axes keep the extents the slices left them. A
    /// boolean mask is `nonzero` in NumPy, an extra index array per axis of
    /// the mask, which phase 0 does not model beside integer indices and so
    /// rejects.
    fn subscript_broadcast(
        &mut self,
        a: &ArrayDesc,
        entries: &[Entry],
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let mut axis = 0usize;
        let mut slots: Vec<Slot> = Vec::with_capacity(entries.len() + a.rank());
        let mut index_dims: Vec<Vec<TermId>> = Vec::new();
        let mut conditions: Vec<crate::phase0::term::BoolId> = Vec::new();
        for (position, entry) in entries.iter().enumerate() {
            match entry {
                Entry::NewAxis => {
                    let one = self.an.terms.constant(1);
                    slots.push(Slot::Kept(one));
                }
                Entry::Slice(index) => {
                    let extent = self.axis_extent(a, axis, position)?;
                    let (lo, hi, step) = self.slice_parts(index)?;
                    let len =
                        slice_extent(&mut self.ctx(), extent, lo.as_ref(), hi.as_ref(), step)?;
                    slots.push(Slot::Kept(len));
                    axis += 1;
                }
                Entry::Scalar(s) => {
                    let extent = self.axis_extent(a, axis, position)?;
                    let t = require_int_term(Rule::EIndex, s)?;
                    conditions.push(in_bounds(&mut self.an.terms, t, extent));
                    index_dims.push(Vec::new());
                    slots.push(Slot::Block);
                    axis += 1;
                }
                Entry::Advanced(indexer) => {
                    self.axis_extent(a, axis, position)?;
                    let base = format!("in_take{}", position);
                    conditions.push(self.ctx().definedness(&base, key));
                    index_dims.push(indexer.dims.clone());
                    slots.push(Slot::Block);
                    axis += 1;
                }
                Entry::Mask { .. } => {
                    return Err(Reject::Unsupported {
                        construct: "a boolean mask beside several integer-array indices"
                            .to_string(),
                    })
                }
            }
        }
        slots.extend(a.dims[axis..].iter().map(|&d| Slot::Kept(d)));
        let refs: Vec<&[TermId]> = index_dims.iter().map(|d| d.as_slice()).collect();
        let (block, compatible) = broadcast_dims(&mut self.an.terms, &refs);
        conditions.push(compatible);
        let defcond = self.an.terms.all(&conditions);
        let dims = place_block(&slots, &block);
        Ok(self
            .ctx()
            .array(Rule::ETakeBroadcast, a.dtype, dims, defcond))
    }

    /// The extent of the axis a subscript entry consumes, or a rejection
    /// naming the entry that has none.
    fn axis_extent(
        &mut self,
        a: &ArrayDesc,
        axis: usize,
        position: usize,
    ) -> Result<TermId, Reject> {
        a.dims
            .get(axis)
            .copied()
            .ok_or_else(|| Reject::Unsupported {
                construct: format!(
                    "subscript entry {} indexes axis {} of a rank-{} array",
                    position,
                    axis,
                    a.rank()
                ),
            })
    }

    /// `E-Index` (§5.4), or `E-Elem` (§5.2) when the array has rank 1.
    fn index_by_scalar(
        &mut self,
        a: &ArrayDesc,
        s: &ScalarDesc,
        axis: usize,
        key: &ValueKey,
        position: usize,
    ) -> Result<Desc, Reject> {
        let base = format!("elem{}", position);
        if a.rank() == 1 && axis == 0 {
            return e_elem(&mut self.ctx(), a, &[s], key, &base);
        }
        e_index(&mut self.ctx(), a, axis, s, key, &base)
    }

    /// A Boolean mask index: `E-Filter` (§5.5). `mask_key` is the value the
    /// mask names, which its count is keyed by.
    fn index_by_mask(
        &mut self,
        a: &ArrayDesc,
        mask: &ArrayDesc,
        mask_key: &ValueKey,
        axis: &mut usize,
        position: usize,
    ) -> Result<Desc, Reject> {
        if *axis != 0 {
            return Err(Reject::Unsupported {
                construct: format!("boolean mask on axis {}", axis),
            });
        }
        let base = format!("c{}", position);
        let filtered = if mask.rank() == a.rank() && a.rank() > 1 {
            e_filter_same(&mut self.ctx(), a, mask, mask_key, &base)?
        } else {
            e_filter_axis0(&mut self.ctx(), a, mask, mask_key, &base)?
        };
        *axis += 1;
        Ok(filtered)
    }

    /// The constant step of a slice (§5.4). A runtime step is out of scope.
    fn constant_step(&mut self, step: Option<&Desc>) -> Result<i64, Reject> {
        let Some(desc) = step else { return Ok(1) };
        let value = self
            .constant_of(desc)
            .ok_or_else(|| Reject::NonConstantStep {
                detail: "a slice step must be a compile-time constant".to_string(),
            })?;
        i64::try_from(value).map_err(|_| Reject::NonConstantStep {
            detail: format!("slice step {} is out of range", value),
        })
    }

    // ── `S-SetItem` (§5.11) ───────────────────────────────────────────

    /// `a[idx] = v`.
    ///
    /// Two keys, in the order §5a fixes: the *write* key, which names the
    /// indexed region — the value `a[idx]` denoted before the write, and the
    /// subject of the index checks — and then one further key for the new
    /// version of the written value, taken after the slice entries exactly as
    /// [`Walker::bind_name`] takes one for an assignment. The name is rebound
    /// to the second.
    ///
    /// The descriptor of an array is unchanged by a write; the rule
    /// contributes the validity of the index — which the read of the region
    /// already records — plus the broadcast compatibility of `v` with it. A
    /// composite instead replaces the element descriptor.
    pub fn setitem(
        &mut self,
        target: &ASTSubscriptAssignTarget,
        value: Desc,
        value_key: Option<&ValueKey>,
    ) -> Result<(), Reject> {
        let write = self.an.keys.next_expr();
        let ((base, base_key, indices, region), frame) = self.in_value(&write, |w| {
            let (base, base_key) = w.expr_keyed(&target.target)?;
            let indices = w.slice_indices(&target.slicing)?;
            let region = w.subscript(&base, &indices, &write)?;
            Ok((base, base_key, indices, region))
        })?;
        self.record(write.clone(), None, &region, frame.into_producer(&region));

        // The write is what the rule's conditions are about (§5a), so the
        // rest runs with the write key as the subject: `check_assignable`
        // records a check, and a composite whose index is a runtime value
        // merges every element.
        let updated = self.with_subject(&write, |w| match &base {
            Desc::Array(_) => {
                // The written array keeps its descriptor: a write changes no
                // extent (§5.11).
                w.check_assignable(&region, &value, &write)?;
                Ok(base.clone())
            }
            Desc::Composite(c) => {
                let c = c.clone();
                w.composite_setitem(&c, &indices, value, &write)
            }
            other => Err(Reject::Unsupported {
                construct: format!("subscript assignment to a {}", other.kind_str()),
            }),
        })?;

        // The new version of the written value, under its own key, and in
        // the state the write happened in (§2).
        let updated = super::created_at(updated, self.path());
        let version = self.an.keys.next_expr();
        let operands = [Some(base_key), Some(write), value_key.cloned()]
            .into_iter()
            .flatten()
            .collect();
        let producer = Producer::Applied {
            rules: vec![Rule::SSetItem],
            operands,
            condition: None,
        };
        self.record(version.clone(), None, &updated, producer);
        self.rebind_subscript_base(&target.target, updated, version)
    }

    /// The broadcast compatibility of the written value with the indexed
    /// region (`S-SetItem`, §5.11).
    fn check_assignable(
        &mut self,
        region: &Desc,
        value: &Desc,
        write: &ValueKey,
    ) -> Result<(), Reject> {
        let region_dims = match region.dims() {
            Some(dims) => dims.to_vec(),
            None => return Ok(()),
        };
        let source = self.as_array(value, Rule::SSetItem)?;
        if source.rank() > region_dims.len() {
            return Err(Reject::Premise {
                rule: Rule::SSetItem,
                detail: format!(
                    "writes a rank-{} value into a rank-{} region",
                    source.rank(),
                    region_dims.len()
                ),
            });
        }
        let (_, defcond) = broadcast_dims(&mut self.an.terms, &[&region_dims, &source.dims]);
        // The check is about the written region, which is the value at the
        // write key (`implementation.md` §5a).
        self.with_subject(write, |w| {
            w.ctx().check(Rule::SSetItem, defcond);
            Ok(())
        })
    }

    /// `v[i] = e` on a composite: the element descriptor is replaced for a
    /// constant `i`, merged for a runtime one (§5.11).
    fn composite_setitem(
        &mut self,
        c: &CompositeDesc,
        indices: &[Index],
        value: Desc,
        key: &ValueKey,
    ) -> Result<Desc, Reject> {
        let [Index::Single { desc: index, .. }] = indices else {
            return Err(Reject::Unsupported {
                construct: "composite element assignment with a compound index".to_string(),
            });
        };
        let Some(scalar) = index.as_scalar() else {
            return Err(Reject::Unsupported {
                construct: "composite element assignment with a non-scalar index".to_string(),
            });
        };
        let length = c.elems.len() as i128;
        let position = scalar
            .term
            .as_int()
            .and_then(|t| self.an.terms.constant_value(t))
            .map(|n| if n < 0 { n + length } else { n });
        let mut elems = c.elems.clone();
        match position {
            Some(at) if at >= 0 && at < length => {
                elems[at as usize] = value;
            }
            Some(at) => {
                return Err(Reject::Unsupported {
                    construct: format!(
                        "index {} out of range for a composite of length {}",
                        at, length
                    ),
                })
            }
            None => {
                // A runtime index may hit any element, so every element
                // becomes the merge of its old value with the written one.
                let hits: Vec<_> = (0..elems.len()).collect();
                for at in hits {
                    let target = self.an.terms.constant(at as i128);
                    let index_term = scalar.term.as_int().ok_or_else(|| Reject::Unsupported {
                        construct: "composite element assignment with an opaque index".to_string(),
                    })?;
                    let hit = self.an.terms.eq(index_term, target);
                    let old = elems[at].clone();
                    elems[at] = merge_desc(
                        &mut self.ctx().in_element(at),
                        hit,
                        &value,
                        &old,
                        key,
                        "setitem",
                    )?;
                }
            }
        }
        Ok(e_composite(c.kind, elems))
    }

    /// Rebind the name a subscript assignment wrote through, to the new
    /// version of the value (§5a).
    fn rebind_subscript_base(
        &mut self,
        base: &ASTNode,
        updated: Desc,
        version: ValueKey,
    ) -> Result<(), Reject> {
        match base {
            ASTNode::ASTLoad(load) => {
                self.frame_mut().set(&load.name, updated, Some(version));
                Ok(())
            }
            other => Err(Reject::Unsupported {
                construct: format!(
                    "element assignment through a {}, which produces a temporary",
                    super::node_name(other)
                ),
            }),
        }
    }
}

/// Replace an `Ellipsis` by the full slices it stands for (§5.4).
fn expand_ellipsis(indices: &[Index], rank: usize) -> Result<Vec<&Index>, Reject> {
    let ellipses = indices
        .iter()
        .filter(|i| matches!(i, Index::Ellipsis))
        .count();
    if ellipses == 0 {
        return Ok(indices.iter().collect());
    }
    if ellipses > 1 {
        return Err(Reject::Unsupported {
            construct: "more than one Ellipsis in a subscript".to_string(),
        });
    }
    let consumed: usize = indices.iter().map(|i| i.axes()).sum();
    let missing = rank.saturating_sub(consumed);
    let mut out: Vec<&Index> = Vec::with_capacity(indices.len() + missing);
    for index in indices {
        if matches!(index, Index::Ellipsis) {
            for _ in 0..missing {
                out.push(&FULL_SLICE);
            }
        } else {
            out.push(index);
        }
    }
    Ok(out)
}

/// `:`, the entry an `Ellipsis` expands into.
static FULL_SLICE: Index = Index::Range {
    lo: None,
    hi: None,
    step: None,
};

/// The scalar behind an optional slice endpoint.
fn as_scalar(d: Option<&Desc>) -> Option<ScalarDesc> {
    d.and_then(|d| d.as_scalar()).cloned()
}
