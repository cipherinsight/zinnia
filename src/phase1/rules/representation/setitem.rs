//! `SETITEM` (`revision/phase1/rules/representation/setitem.md`).
//!
//! An in-place write as a new version `a′` of `a`: every slot of `a`'s
//! bundle gets one `select` that takes the written value when the slot's
//! coordinate lies in the written region R and the old value otherwise;
//! validity, keys, shape and ℓ are `a`'s wires by alias. There is no
//! writable memory, so this per-slot select is the only form.
//!
//! **What the rule reads.** Phase 0 records a write as two keys: the *write*
//! key, which names the region `a[idx]` as it was read, and the new version,
//! whose producer is `Applied(S-SetItem, [a, write, v])`
//! (`../../../phase0/definition.md` §5.11). The subscript's entries are on
//! the AST node of the write key and the region's extents on its descriptor,
//! so the relative coordinate ρ of a slot is read off Δ and not recomputed.
//!
//! **Which forms are written.** Over a dense `old` the slot's coordinate is
//! the constant p, so every entry kind of the rule file applies: an integer
//! index, a slice with a constant step, an axis the subscript does not
//! mention, one Boolean mask, one index array. Over a ranked or keyed `old`
//! only the mask entry is written, as the rule file's `premises` say: ρ is
//! then not a term, so no `fetched` form exists there either.

use crate::phase0::desc::{Desc, Dtype, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::{BoolId, TermId};

use crate::phase1::ast_index::SubscriptEntry;
use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, BatchEntry, Capacity, Coordinate, Family, Interface, Layout, LayoutKind, Live,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct SetItem;

/// One entry of the subscript, as `setitem.md`'s table classifies it.
#[derive(Debug, Clone)]
enum Entry {
    /// I: an integer scalar, with the normalised position and the extent.
    Index { position: TermId, extent: TermId, index: Target },
    /// S: `lo:hi:step` with a constant step.
    Slice { lower: TermId, upper: TermId, step: i64 },
    /// An axis the subscript does not restrict.
    Full,
    /// M: a Boolean mask, which stands alone.
    Mask(ValueKey),
    /// F: an integer index array on axis 0, which stands alone.
    Fancy(ValueKey),
}

/// What the source pattern read.
struct Matched {
    old: ValueKey,
    /// The region `a[idx]` as it was read: its descriptor gives the extents
    /// the written value must match.
    region: ValueKey,
    value: Option<ValueKey>,
    entries: Vec<Entry>,
}

impl Rule for SetItem {
    fn id(&self) -> RuleId {
        RuleId::SetItem
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
        let Some(matched) = self.matches(builder, &key)? else {
            return Ok(Vec::new());
        };
        let Some(array) = builder.array(&key).cloned() else {
            return Ok(Vec::new());
        };
        if !super::super::sources::encoded(builder, "SETITEM", array.dtype) {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        // `old`: every concrete bundle the row of a offers, in the family the
        // row asks for — a′ is held exactly as a is.
        let psi = builder.psi(&matched.old);
        let mut probe = Candidate::new(RuleId::SetItem, None, request.epoch);
        let condition = builder.child_under_g(&mut probe, request, psi, &matched.old.encode())?;
        let row = builder.child(
            request,
            Target::Value(matched.old.clone()),
            request.family,
            condition,
        )?;
        for entry in &row.entries.clone() {
            let Some(lambda) = entry.interface.lambda().cloned() else {
                continue;
            };
            out.extend(self.over(builder, request, &key, &array, &matched, entry, &lambda)?);
        }
        Ok(out)
    }
}

impl SetItem {
    /// `source_pattern`: `Applied(S-SetItem, [a, write, v])` with the
    /// subscript on the write key's AST node.
    fn matches(
        &self,
        builder: &mut Builder<'_>,
        key: &ValueKey,
    ) -> Result<Option<Matched>, Reject> {
        let Some(entry) = builder.entry(key) else {
            return Ok(None);
        };
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return Ok(None);
        };
        if !rules.iter().all(|rule| *rule == Phase0Rule::SSetItem) || operands.len() < 2 {
            return Ok(None);
        }
        let (old, region) = (operands[0].clone(), operands[1].clone());
        let value = operands.get(2).cloned();
        if builder.array(&old).is_none() {
            return Ok(None);
        }
        let Some(entries) = self.entries(builder, &old, &region) else {
            return Ok(None);
        };
        Ok(Some(Matched { old, region, value, entries }))
    }

    /// The per-axis entries of the subscript, normalised to one per axis of
    /// `a` (`setitem.md`, `source_pattern`).
    fn entries(
        &self,
        builder: &mut Builder<'_>,
        old: &ValueKey,
        region: &ValueKey,
    ) -> Option<Vec<Entry>> {
        let rank = builder.array(old)?.rank();
        let dims = builder.array(old)?.dims.clone();
        let region_dims = builder.array(region).map(|array| array.dims.clone());
        let written = builder.node(region)?.entries()?.to_vec();
        // An ellipsis stands for the axes the other entries do not name, so
        // it expands to that many `full` entries (`setitem.md`, the `full`
        // row). Two ellipses name no axis set at all.
        let ellipses = written
            .iter()
            .filter(|entry| matches!(entry, SubscriptEntry::Ellipsis))
            .count();
        if ellipses > 1 {
            return None;
        }
        let named = written.len() - ellipses;
        let mut out: Vec<Entry> = Vec::new();
        // The axis of `a` the entry restricts, and the axis of the region
        // the entry's own axes start at; an integer entry drops its axis from
        // the region and an index array contributes its own rank.
        let mut axis = 0usize;
        let mut output_axis = 0usize;
        for entry in written.iter() {
            if matches!(entry, SubscriptEntry::Ellipsis) {
                for _ in 0..rank.checked_sub(named)? {
                    out.push(Entry::Full);
                    axis += 1;
                    output_axis += 1;
                }
                continue;
            }
            if axis >= rank {
                return None;
            }
            match entry {
                SubscriptEntry::Slice { lo, hi, step } => {
                    let step = step.unwrap_or(1);
                    let extent = *region_dims.as_ref()?.get(output_axis)?;
                    // The endpoints are read back out of the region's length
                    // term, which phase 0 built over them
                    // (`mapping/README.md`). A slice with constant endpoints
                    // has a folded length — `b[1:3]` is the literal 2 — and
                    // then the two are the normalised endpoints of the
                    // written entry, built here as phase 0 builds them.
                    let (lower, upper) =
                        match crate::phase1::rules::mapping::slice_bounds(builder, extent, step) {
                            Some(pair) => pair,
                            None => self.endpoints(builder, region, *lo, *hi, dims[axis], step)?,
                        };
                    out.push(Entry::Slice { lower, upper, step });
                    axis += 1;
                    output_axis += 1;
                }
                SubscriptEntry::Single { expr_index } => {
                    let sibling = builder.sibling(region, *expr_index)?;
                    let target = Target::Value(sibling.key.clone());
                    match sibling.desc.clone() {
                        Desc::Scalar(scalar) => {
                            let term = scalar.term.as_int()?;
                            let position = self.normalised(builder, term, dims[axis]);
                            out.push(Entry::Index {
                                position,
                                extent: dims[axis],
                                index: target,
                            });
                            // An integer entry drops its axis from the
                            // region, so `output_axis` does not advance.
                            axis += 1;
                        }
                        // A Boolean index is a mask, which stands alone
                        // (`setitem.md`).
                        Desc::Array(array) if array.dtype == Dtype::Bool => {
                            if written.len() != 1 {
                                return None;
                            }
                            out.push(Entry::Mask(builder.defining_key(&sibling.key.clone())));
                            axis += 1;
                        }
                        // F: an index array, on any axis and beside the
                        // integer, slice and full entries of the other axes
                        // (`setitem.md`, the F row). It contributes its own
                        // rank to the region.
                        Desc::Array(index) => {
                            out.push(Entry::Fancy(sibling.key.clone()));
                            axis += 1;
                            output_axis += index.rank();
                        }
                        _ => return None,
                    }
                }
                // A `None` entry or an ellipsis in a write target is outside
                // the rule file's table.
                _ => return None,
            }
        }
        while out.len() < rank {
            out.push(Entry::Full);
        }
        Some(out)
    }

    /// `(L, U)` of a slice entry whose length term folded: the normalised
    /// endpoints `norm(t, d)` of `../../../phase0/definition.md` §5.4, built
    /// through phase 0's own helper, with Python's defaults for an omitted
    /// endpoint.
    fn endpoints(
        &self,
        builder: &mut Builder<'_>,
        region: &ValueKey,
        lo: Option<u32>,
        hi: Option<u32>,
        extent: TermId,
        step: i64,
    ) -> Option<(TermId, TermId)> {
        let written = |builder: &mut Builder<'_>, at: Option<u32>| -> Option<TermId> {
            let term = builder
                .sibling(region, at?)
                .and_then(|entry| entry.desc.as_scalar())
                .and_then(|scalar| scalar.term.as_int())?;
            Some(crate::phase0::rules::helpers::norm(
                &mut builder.terms,
                term,
                extent,
                step < 0,
            ))
        };
        let lower = match written(builder, lo) {
            Some(term) => term,
            // An omitted start is 0 going up and `d − 1` going down.
            None if step > 0 => builder.literal(0),
            None => {
                let one = builder.literal(1);
                builder.terms.sub(extent, one)
            }
        };
        let upper = match written(builder, hi) {
            Some(term) => term,
            None if step > 0 => extent,
            None => builder.literal(-1),
        };
        Some((lower, upper))
    }

    /// `j = ite(t < 0, t + d, t)`, the normalised position of an integer
    /// entry (`setitem.md`, the I row).
    fn normalised(&self, builder: &mut Builder<'_>, term: TermId, extent: TermId) -> TermId {
        let zero = builder.literal(0);
        let negative = builder.terms.lt(term, zero);
        let wrapped = builder.terms.add(extent, term);
        builder.terms.ite(negative, wrapped, term)
    }

    /// The nodes one bundle of `a` gives.
    #[allow(clippy::too_many_arguments)]
    fn over(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        matched: &Matched,
        old: &crate::phase1::table::RowEntry,
        lambda: &Layout,
    ) -> Result<Vec<Candidate>, Reject> {
        let slots = lambda.layout.slots();
        let dense = lambda.layout.is_dense();
        let grid = match &lambda.layout {
            LayoutKind::Box(grid) => grid.clone(),
            _ => vec![slots],
        };
        let mask = matched.entries.iter().find_map(|entry| match entry {
            Entry::Mask(mask) => Some(mask.clone()),
            _ => None,
        });
        // Over a ranked or keyed `old` only the mask entry is written
        // (`setitem.md`, `premises`).
        if !dense && mask.is_none() {
            return Ok(Vec::new());
        }
        // A per-axis entry tests the slot's coordinate on its own axis, which
        // only a bundle with one capacity per axis of a carries: a compact
        // bundle of a rank-≥2 value holds the C-order position instead, and
        // the region is then not the product of per-axis tests.
        let per_axis = matched
            .entries
            .iter()
            .any(|entry| !matches!(entry, Entry::Mask(_) | Entry::Fancy(_)));
        if per_axis && grid.len() != matched.entries.len() {
            return Ok(Vec::new());
        }
        let mut candidate = Candidate::new(RuleId::SetItem, None, request.epoch);
        candidate.matched.push(key.clone());
        candidate.inputs.push(old.requirement);
        candidate.epoch = candidate.epoch.max(old.epoch);
        let gated = !builder.terms.is_true(request.condition);
        if gated {
            let gate = builder.child(
                request,
                Target::BoolTerm(request.condition),
                Family::Wire,
                builder.terms.tt(),
            )?;
            let Some(first) = gate.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }

        // The region bits, with the ports each entry kind raises.
        let Some((mut candidate, ranks)) =
            self.region(builder, request, &mut candidate.clone(), matched, lambda, &grid)?
        else {
            return Ok(Vec::new());
        };
        // The written value, one variant per way of reading it.
        let mut out = Vec::new();
        for mut written in self.written(
            builder,
            request,
            &mut candidate,
            matched,
            lambda,
            &grid,
            ranks,
        )? {
            // Update: one `select` per slot.
            written = written
                .emit(Primitive::Select, slots)
                .output(
                    request.target.clone(),
                    self.bundle(array, key, lambda, old, slots),
                    request.condition,
                )
                .parameter("Q", slots)
                .parameter("w", bits_of_capacity(slots) as i128);
            out.push(written);
        }
        Ok(out)
    }

    /// The bundle `a′` is held in: `old`'s interface with `a′` as the owner
    /// and `a`'s own λ, so that the two versions are aligned.
    fn bundle(
        &self,
        array: &crate::phase0::desc::ArrayDesc,
        key: &ValueKey,
        lambda: &Layout,
        old: &crate::phase1::table::RowEntry,
        slots: Capacity,
    ) -> Interface {
        match &old.interface {
            Interface::Masked { .. } => Interface::Masked {
                dtype: array.dtype,
                capacity: slots,
                lambda: lambda.clone(),
            },
            Interface::Packed { canon, .. } => Interface::Packed {
                dtype: array.dtype,
                capacity: slots,
                lambda: lambda.clone(),
                canon: *canon,
            },
            _ => Interface::Store(crate::phase1::interface::StoreIface {
                owner: key.clone(),
                dtype: array.dtype,
                capacity: slots,
                lambda: lambda.clone(),
            }),
        }
    }

    /// The region bits `in_R(p)`: the ports the entries raise and the gates
    /// they cost. Returns the `Ranks` requirement of a mask entry, which the
    /// `routed` and `aligned` forms read.
    #[allow(clippy::too_many_arguments)]
    fn region(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        candidate: &mut Candidate,
        matched: &Matched,
        lambda: &Layout,
        grid: &[Capacity],
    ) -> Result<Option<(Candidate, Option<crate::phase1::table::RowEntry>)>, Reject> {
        let mut node = candidate.clone();
        let slots: Capacity = grid.iter().product();
        let dense = lambda.layout.is_dense();
        let under = builder.under(request.condition);
        let mut restricted = 0i128;
        let mut ranks = None;
        for (axis, entry) in matched.entries.iter().enumerate() {
            let extent = grid.get(axis).copied().unwrap_or(slots);
            match entry {
                Entry::Full => {}
                Entry::Index { position, index, .. } => {
                    restricted += 1;
                    let answer = builder.ask_for(
                        &mut node,
                        Question::Const(*position),
                        request.epoch,
                        under,
                    );
                    if matches!(answer, Verdict::Constant(_)) {
                        continue;
                    }
                    // `idx_k`: the normalised position, from `INDEX_NORMALIZE`.
                    let row = builder.index_norm_port(
                        request,
                        index.clone(),
                        matched
                            .entries
                            .get(axis)
                            .and_then(|entry| match entry {
                                Entry::Index { extent, .. } => Some(*extent),
                                _ => None,
                            })
                            .unwrap_or(*position),
                        request.condition,
                    )?;
                    let Some(first) = row.entries.first() else {
                        return Ok(None);
                    };
                    node.inputs.push(first.requirement);
                    node.epoch = node.epoch.max(first.epoch);
                    node = node.emit(Primitive::OneHot { capacity: extent }, 1);
                }
                Entry::Slice { lower, upper, step } => {
                    restricted += 1;
                    for &term in [lower, upper].iter() {
                        let answer = builder.ask_for(
                            &mut node,
                            Question::Const(*term),
                            request.epoch,
                            under,
                        );
                        if matches!(answer, Verdict::Constant(_)) {
                            continue;
                        }
                        let row = builder.child(
                            request,
                            Target::Term(*term),
                            Family::Wire,
                            request.condition,
                        )?;
                        let Some(first) = row.entries.first() else {
                            return Ok(None);
                        };
                        node.inputs.push(first.requirement);
                        node.epoch = node.epoch.max(first.epoch);
                    }
                    // Two thermometers and one `and` per position, with a
                    // stride test when the step is not ±1.
                    node = node
                        .emit(Primitive::Therm { capacity: extent }, 2)
                        .emit(Primitive::And, extent);
                    if step.abs() != 1 {
                        node = node
                            .emit(Primitive::OneHot { capacity: step.unsigned_abs() as i128 }, 1)
                            .emit(Primitive::And, extent);
                    }
                }
                Entry::Mask(mask) => {
                    restricted += 1;
                    // `sel`: the `Ranks(I_m)` node of a bundle of m aligned
                    // with `old`, which gives both the region bits and the
                    // ranks the `routed` form needs.
                    let Some(found) = self.selection(builder, request, &mut node, mask, lambda)?
                    else {
                        return Ok(None);
                    };
                    node.inputs.push(found.requirement);
                    node.epoch = node.epoch.max(found.epoch);
                    ranks = Some(found);
                }
                Entry::Fancy(index) => {
                    restricted += 1;
                    let Some(dims) = builder.array(index).map(|array| array.dims.clone()) else {
                        return Ok(None);
                    };
                    let boxes = builder.capacity_box(&mut node, &dims, request.epoch, under);
                    let Some(capacities) = boxes.first().cloned() else {
                        return Ok(None);
                    };
                    let batch = Batch::grid(&capacities.axes);
                    let row = builder.fetch_port(&mut node, request, index, &batch)?;
                    let Some(first) = row.entries.first() else {
                        return Ok(None);
                    };
                    node.inputs.push(first.requirement);
                    node.epoch = node.epoch.max(first.epoch);
                    let indices = capacities.slots();
                    let width = builder.omega.round(builder.dtype_width(Dtype::I64));
                    node = node
                        .emit(Primitive::Cmp { width }, indices)
                        .emit(Primitive::OneHot { capacity: extent }, indices)
                        .emit(Primitive::Mul, indices * extent);
                }
            }
        }
        // `in_R(p) = ⋀_k in_k(p_k)`, one `and` per restricted axis beyond the
        // first, and one more for the gate.
        node = node
            .emit(Primitive::And, (restricted - 1).max(0) * slots)
            .emit(Primitive::And, i128::from(!builder.terms.is_true(request.condition)) * slots);
        if !dense {
            // On a ranked or keyed old every test is against the slot's key
            // and is conjoined with the slot's validity.
            node = node.emit(Primitive::And, slots);
        }
        Ok(Some((node, ranks)))
    }

    /// The `sel` port of a mask entry: the `Ranks(I_m)` node of a bundle of m
    /// aligned with `old` (§4.1).
    fn selection(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        candidate: &mut Candidate,
        mask: &ValueKey,
        lambda: &Layout,
    ) -> Result<Option<crate::phase1::table::RowEntry>, Reject> {
        let Some(mask_dims) = builder.array(mask).map(|array| array.dims.clone()) else {
            return Ok(None);
        };
        let old_dims = mask_dims.clone();
        for family in [Family::Store, Family::Masked, Family::Packed] {
            let psi = builder.psi(mask);
            let condition = builder.child_under_g(candidate, request, psi, &mask.encode())?;
            let row = builder.child(request, Target::Value(mask.clone()), family, condition)?;
            for entry in &row.entries.clone() {
                let Some(mask_lambda) = entry.interface.lambda().cloned() else {
                    continue;
                };
                let under = builder.under(request.condition);
                if !builder.aligned(
                    candidate,
                    lambda,
                    &mask_lambda,
                    &old_dims,
                    &mask_dims,
                    request.epoch,
                    under,
                ) {
                    continue;
                }
                let Some(subject) = super::super::masks::subject_of(&entry.interface, mask, false)
                else {
                    continue;
                };
                let ranks = builder.ranks_port(candidate, request, mask, subject)?;
                if let Some(first) = ranks.entries.first() {
                    return Ok(Some(first.clone()));
                }
            }
        }
        Ok(None)
    }

    /// The written value: one candidate per variant of `setitem.md` that the
    /// ports admit.
    #[allow(clippy::too_many_arguments)]
    fn written(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        candidate: &mut Candidate,
        matched: &Matched,
        lambda: &Layout,
        grid: &[Capacity],
        ranks: Option<crate::phase1::table::RowEntry>,
    ) -> Result<Vec<Candidate>, Reject> {
        let slots: Capacity = grid.iter().product();
        let Some(value) = matched.value.clone() else {
            // A literal right-hand side carries no key of Δ: its wire is a
            // constant the `select` folds in, and the node has no `val` port.
            let mut node = candidate.clone();
            node.variant = Some("scalar");
            return Ok(vec![node]);
        };
        // premises: an array v is written position by position, so every
        // axis of v must have the region's extent — broadcasting is
        // S-SetItem's defcond and this rule asks the stronger equality
        // (`setitem.md`, `premises`). An axis of extent 1 broadcasts and is
        // admitted.
        if builder.array(&value).is_some() && !self.conformable(builder, request, candidate, matched)
        {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        // `scalar`: the written value is one wire on the whole region.
        if builder.scalar(&value).is_some() {
            let mut node = candidate.clone();
            node.variant = Some("scalar");
            let psi = builder.psi(&value);
            let condition = builder.child_under_g(&mut node, request, psi, &value.encode())?;
            let row = builder.child(request, Target::Value(value.clone()), Family::Wire, condition)?;
            if let Some(first) = row.entries.first() {
                node.inputs.push(first.requirement);
                node.epoch = node.epoch.max(first.epoch);
                out.push(node);
            }
            return Ok(out);
        }
        // An index array with an array right-hand side is `setitem.md`'s
        // Open: the region is the product of F on its axis with the other
        // entries, but which element of v a slot takes — the block v carries
        // per index — is not written, so only the `scalar` form above is
        // offered here.
        if matched
            .entries
            .iter()
            .any(|entry| matches!(entry, Entry::Fancy(_)))
        {
            return Ok(Vec::new());
        }
        // `routed`: the element of rank r_p, from a `Packed` of v, routed to
        // slot p by a one-hot. Only a mask entry has the ranks it needs.
        if let Some(ranks) = &ranks {
            let psi = builder.psi(&value);
            let mut node = candidate.clone();
            node.variant = Some("routed");
            let condition = builder.child_under_g(&mut node, request, psi, &value.encode())?;
            let row =
                builder.child(request, Target::Value(value.clone()), Family::Packed, condition)?;
            for entry in &row.entries {
                let Interface::Packed { capacity, .. } = &entry.interface else {
                    continue;
                };
                let mut node = node.clone();
                node.inputs.push(entry.requirement);
                node.epoch = node.epoch.max(entry.epoch);
                let _ = ranks;
                out.push(
                    node.emit(Primitive::OneHot { capacity: *capacity }, slots)
                        .emit(Primitive::Mul, slots * *capacity),
                );
            }
        }
        // `fetched`: v read at the relative coordinates of every slot.
        if lambda.layout.is_dense() {
            let mut node = candidate.clone();
            node.variant = Some("fetched");
            if let Some(batch) = self.value_batch(builder, matched, grid)? {
                let row = builder.fetch_port(&mut node, request, &value, &batch)?;
                if let Some(first) = row.entries.first() {
                    node.inputs.push(first.requirement);
                    node.epoch = node.epoch.max(first.epoch);
                    out.push(node);
                }
            }
        }
        Ok(out)
    }

    /// `PROVE(dims_v[a] = the region's extent on that axis)` per axis, with
    /// an axis of constant extent 1 admitted as a broadcast
    /// (`setitem.md`, `premises`).
    fn conformable(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        candidate: &mut Candidate,
        matched: &Matched,
    ) -> bool {
        let Some(value) = matched.value.clone() else {
            return false;
        };
        let (Some(dims), Some(region)) = (
            builder.array(&value).map(|array| array.dims.clone()),
            builder.array(&matched.region).map(|array| array.dims.clone()),
        ) else {
            return false;
        };
        if dims.len() > region.len() {
            return false;
        }
        // v is aligned with the trailing axes of the region, as NumPy
        // broadcasts.
        let offset = region.len() - dims.len();
        let under = builder.under(request.condition);
        dims.iter().enumerate().all(|(axis, &extent)| {
            if builder.terms.constant_value(extent) == Some(1) {
                return true;
            }
            extent == region[axis + offset]
                || builder.prove_eq(
                    candidate,
                    extent,
                    region[axis + offset],
                    request.epoch,
                    under,
                )
        })
    }

    /// The batch the `fetched` form reads v at: one entry per slot of `old`,
    /// with the relative coordinate ρ(p) and the live term `p ∈ R`
    /// (`setitem.md`, `required_input_ports`).
    ///
    /// An axis of v whose extent is the literal 1 broadcasts: its coordinate
    /// is 0. An axis whose extent is undecided takes the `ite` coordinate
    /// `ite(dims_v[a] = 1, 0, ρ_a)`, which is a term like any other.
    fn value_batch(
        &self,
        builder: &mut Builder<'_>,
        matched: &Matched,
        grid: &[Capacity],
    ) -> Result<Option<Batch>, Reject> {
        let Some(value) = matched.value.clone() else {
            return Ok(None);
        };
        let Some(dims) = builder.array(&value).map(|array| array.dims.clone()) else {
            return Ok(None);
        };
        // The axes of the region ρ runs over: every entry but an integer one.
        let axes: Vec<usize> = matched
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !matches!(entry, Entry::Index { .. }))
            .map(|(axis, _)| axis)
            .collect();
        if axes.len() != dims.len() {
            return Ok(None);
        }
        let strides = strides_of(grid);
        let mut entries = Vec::with_capacity(grid.iter().product::<Capacity>().max(0) as usize);
        for slot in 0..grid.iter().product::<Capacity>() {
            let mut coordinate = Vec::with_capacity(axes.len());
            let mut live: Vec<BoolId> = Vec::new();
            for (at, &axis) in axes.iter().enumerate() {
                let position = (slot / strides[axis]) % grid[axis];
                let relative = match &matched.entries[axis] {
                    Entry::Slice { lower, step, .. } => {
                        let base = builder.literal(position);
                        let shifted = builder.terms.sub(base, *lower);
                        match builder.terms.div_const(shifted, (*step).abs() as i128) {
                            Some(term) => term,
                            None => return Ok(None),
                        }
                    }
                    _ => builder.literal(position),
                };
                // A broadcast axis reads position 0.
                let one = builder.literal(1);
                let singleton = builder.terms.eq(dims[at], one);
                let zero = builder.literal(0);
                let coordinate_term = builder.terms.ite(singleton, zero, relative);
                coordinate.push(Coordinate::Term(coordinate_term));
            }
            // `in_R(p)`: the slice bounds of every restricted axis, as a term.
            for (axis, entry) in matched.entries.iter().enumerate() {
                let position = (slot / strides[axis]) % grid[axis];
                match entry {
                    Entry::Slice { lower, upper, .. } => {
                        let base = builder.literal(position);
                        let above = builder.terms.le(*lower, base);
                        let below = builder.terms.lt(base, *upper);
                        live.push(above);
                        live.push(below);
                    }
                    Entry::Index { position: term, .. } => {
                        let base = builder.literal(position);
                        let at = builder.terms.eq(*term, base);
                        live.push(at);
                    }
                    _ => {}
                }
            }
            let live = crate::phase1::build::conjunction(&mut builder.terms, &live);
            entries.push(BatchEntry { coordinate, live: Live::Term(live) });
        }
        Ok(Some(Batch::General(entries)))
    }
}

/// The C-order strides of a capacity grid.
fn strides_of(grid: &[Capacity]) -> Vec<Capacity> {
    let mut strides = vec![1 as Capacity; grid.len()];
    for axis in (0..grid.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * grid[axis + 1];
    }
    strides
}
