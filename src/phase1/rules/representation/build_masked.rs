//! `BUILD_MASKED` (`revision/phase1/rules/representation/build_masked.md`).
//!
//! Holds a value in place: the elements of `v = x[m]`, or of
//! `v = u[lo:hi]`, stay in the slots of the base bundle and one validity bit
//! per slot says which slots belong to v. No element moves and no payload is
//! allocated; the node's own wires are the validity bits and its cost is
//! what computes them.
//!
//! Three variants, as the rule file writes them: `filter-aligned` takes the
//! selection bits from the `Ranks(I_m)` node of a bundle of m aligned with
//! the base, at no gate; `filter-fetched` reads m at the base's constant
//! coordinates and gates each slot, over a dense base; `slice` cuts the
//! base's validity at the two normalised endpoints.
//!
//! The capacity is the base's: holding in place keeps the capacity of what
//! it selects from, so the rule asks no MAX.
//!
//! **Rank.** The rank-1 forms produce `Masked(v, B; λ)` and E-Filter-Axis0
//! produces a rank-≥2 `Store` ranked over the base's C-order box. The
//! `slice` variant is written for a rank-1 base only: at rank ≥ 2 the cut is
//! on axis 0 while the prefix chain runs over the whole box, which the rule
//! file's emitter does not state.

use crate::phase0::desc::ValueKey;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Batch, Capacity, Family, Interface, Layout, LayoutKind, LineageStep, StoreIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct BuildMasked;

/// What the source pattern read: a filter or a step-1 slice on axis 0.
enum Matched {
    /// `v = x[m]`, one of the three filter rules.
    Filter { base: ValueKey, mask: ValueKey },
    /// `v = u[lo:hi]` with step +1 on axis 0.
    Slice { base: ValueKey, lower: TermId, upper: TermId },
}

impl Matched {
    fn base(&self) -> &ValueKey {
        match self {
            Matched::Filter { base, .. } | Matched::Slice { base, .. } => base,
        }
    }
}

/// The families a base bundle may be held in.
const BASES: [Family; 3] = [Family::Store, Family::Masked, Family::Packed];

impl Rule for BuildMasked {
    fn id(&self) -> RuleId {
        RuleId::BuildMasked
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Masked, Family::Store]
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
        if array.rank() == 0 || !super::super::sources::encoded(builder, "BUILD_MASKED", array.dtype)
        {
            return Ok(Vec::new());
        }
        // The rank-1 bundle is a `Masked`, the rank-≥2 one a ranked `Store`.
        let wanted = if array.rank() == 1 { Family::Masked } else { Family::Store };
        if request.family != wanted {
            return Ok(Vec::new());
        }
        let base = matched.base().clone();
        let Some(base_dims) = builder.array(&base).map(|array| array.dims.clone()) else {
            return Ok(Vec::new());
        };

        let mut out = Vec::new();
        for family in BASES {
            let mut probe = Candidate::new(RuleId::BuildMasked, None, request.epoch);
            let psi = builder.psi(&base);
            let condition = builder.child_under_g(&mut probe, request, psi, &base.encode())?;
            let row = builder.child(request, Target::Value(base.clone()), family, condition)?;
            for entry in &row.entries.clone() {
                let Some(lambda) = entry.interface.lambda().cloned() else {
                    continue;
                };
                // premises (structural): the base's layout is ordered.
                if !lambda.layout.is_ordered() {
                    continue;
                }
                let slots = lambda.layout.slots();
                match &matched {
                    Matched::Filter { mask, .. } => {
                        out.extend(self.aligned(
                            builder, request, &key, &array, mask, entry, &lambda, &base_dims,
                        )?);
                        out.extend(self.fetched(
                            builder, request, &key, &array, mask, entry, &lambda, slots,
                        )?);
                    }
                    Matched::Slice { lower, upper, .. } => {
                        out.extend(self.sliced(
                            builder, request, &key, &array, entry, &lambda, *lower, *upper,
                        )?);
                    }
                }
            }
        }
        Ok(out)
    }
}

impl BuildMasked {
    /// `source_pattern`: `Applied(E-Filter-*, [x, m])`, or `Applied(E-Slice,
    /// [u, …])` whose subscript has step +1 on axis 0 and a run-time
    /// endpoint.
    fn matches(&self, builder: &Builder<'_>, key: &ValueKey) -> Option<Matched> {
        let entry = builder.entry(key)?;
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return None;
        };
        // Every rule of the producer must be a filter: a value that was
        // filtered and then reshaped is one key with two rules, and the
        // bundle this rule would build is of the intermediate value Δ never
        // keyed.
        if !rules.is_empty()
            && rules.iter().all(|rule| {
                matches!(
                    rule,
                    Phase0Rule::EFilter1D | Phase0Rule::EFilterAxis0 | Phase0Rule::EFilterSame
                )
            })
        {
            let arrays: Vec<&ValueKey> = operands
                .iter()
                .filter(|operand| builder.array(operand).is_some())
                .collect();
            let base = (*arrays.first()?).clone();
            let mask = builder.defining_key(arrays.get(1)?);
            return Some(Matched::Filter { base, mask });
        }
        if !rules.contains(&Phase0Rule::ESlice) || rules.len() != 1 {
            return None;
        }
        // A step other than +1, or an entry on another axis, is the mapping
        // group's: this variant keeps the base's slots and cuts its validity.
        let node = builder.node(key)?;
        let entries = node.entries()?;
        let step = match entries.first()? {
            crate::phase1::ast_index::SubscriptEntry::Slice { step, .. } => step.unwrap_or(1),
            _ => return None,
        };
        if step != 1 || builder.array(key)?.rank() != 1 {
            return None;
        }
        let base = operands
            .iter()
            .find(|operand| builder.array(operand).is_some())?
            .clone();
        let (lower, upper) =
            crate::phase1::rules::mapping::slice_bounds(builder, builder.array(key)?.dims[0], 1)?;
        Some(Matched::Slice { base, lower, upper })
    }

    /// `filter-aligned`: the selection bits are the `t` wires of a
    /// `Ranks(I_m)` node over a bundle of m aligned with the base, so the
    /// output validity is that vector by alias and the node costs nothing.
    #[allow(clippy::too_many_arguments)]
    fn aligned(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        mask: &ValueKey,
        base: &crate::phase1::table::RowEntry,
        lambda: &Layout,
        base_dims: &[TermId],
    ) -> Result<Vec<Candidate>, Reject> {
        let mask_dims = match builder.array(mask) {
            Some(desc) => desc.dims.clone(),
            None => return Ok(Vec::new()),
        };
        let mut out = Vec::new();
        for family in BASES {
            let mut probe = Candidate::new(RuleId::BuildMasked, Some("filter-aligned"), request.epoch);
            let psi = builder.psi(mask);
            let condition = builder.child_under_g(&mut probe, request, psi, &mask.encode())?;
            let row = builder.child(request, Target::Value(mask.clone()), family, condition)?;
            for entry in &row.entries.clone() {
                let Some(mask_lambda) = entry.interface.lambda().cloned() else {
                    continue;
                };
                let mut candidate =
                    Candidate::new(RuleId::BuildMasked, Some("filter-aligned"), request.epoch);
                candidate.matched.push(key.clone());
                // The premise: I_m is aligned with the base (§4.1), decided
                // by construction with one `PROVE(dims equal)` per axis.
                let under = builder.under(request.condition);
                if !builder.aligned(
                    &mut candidate,
                    lambda,
                    &mask_lambda,
                    base_dims,
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
                let ranks = builder.ranks_port(&mut candidate, request, mask, subject)?;
                let Some(first) = ranks.entries.first() else {
                    continue;
                };
                candidate.inputs.push(base.requirement);
                candidate.epoch = candidate.epoch.max(base.epoch);
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
                let Some(candidate) = self.finish(
                    builder,
                    request,
                    key,
                    array,
                    candidate,
                    lambda,
                    LineageStep::Filter { mask: mask.clone() },
                    Some(mask.clone()),
                )?
                else {
                    continue;
                };
                // A `Packed` base contributes the thermometer of its ℓ,
                // which is the validity the aligned bits are gated by.
                let slots = lambda.layout.slots();
                out.push(match base.interface {
                    Interface::Packed { canon: false, .. } => {
                        candidate.emit(Primitive::Therm { capacity: slots }, 1)
                    }
                    _ => candidate,
                });
            }
        }
        Ok(out)
    }

    /// `filter-fetched`: over a dense base, m is read at the base's constant
    /// coordinates and one `and` per slot combines it with the base's
    /// validity.
    #[allow(clippy::too_many_arguments)]
    fn fetched(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        mask: &ValueKey,
        base: &crate::phase1::table::RowEntry,
        lambda: &Layout,
        slots: Capacity,
    ) -> Result<Vec<Candidate>, Reject> {
        if !lambda.layout.is_dense() {
            return Ok(Vec::new());
        }
        let mask_rank = builder.array(mask).map(|desc| desc.rank()).unwrap_or(0);
        if mask_rank == 0 {
            return Ok(Vec::new());
        }
        // The batch: the grid of the base's box for a rank-1 or same-shape
        // mask, and of its leading axis for E-Filter-Axis0.
        let grid = match &lambda.layout {
            LayoutKind::Box(capacities) => {
                if mask_rank == capacities.len() {
                    capacities.clone()
                } else {
                    vec![capacities[0]]
                }
            }
            LayoutKind::Compact(capacity) => vec![*capacity],
            _ => return Ok(Vec::new()),
        };
        let batch = Batch::grid(&grid);
        let mut candidate =
            Candidate::new(RuleId::BuildMasked, Some("filter-fetched"), request.epoch);
        candidate.matched.push(key.clone());
        let values = builder.fetch_port(&mut candidate, request, mask, &batch)?;
        let Some(first) = values.entries.first() else {
            return Ok(Vec::new());
        };
        candidate.inputs.push(base.requirement);
        candidate.epoch = candidate.epoch.max(base.epoch);
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
        let Some(candidate) = self.finish(
            builder,
            request,
            key,
            array,
            candidate,
            lambda,
            LineageStep::Filter { mask: mask.clone() },
            Some(mask.clone()),
        )?
        else {
            return Ok(Vec::new());
        };
        // emitter: `W_i = valid_base,i ∧ y_i`, one `and` per slot.
        Ok(vec![candidate.emit(Primitive::And, slots)])
    }

    /// `slice`: the base's validity cut at the ranks L and U.
    #[allow(clippy::too_many_arguments)]
    fn sliced(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        base: &crate::phase1::table::RowEntry,
        lambda: &Layout,
        lower: TermId,
        upper: TermId,
    ) -> Result<Vec<Candidate>, Reject> {
        let mut candidate = Candidate::new(RuleId::BuildMasked, Some("slice"), request.epoch);
        candidate.matched.push(key.clone());
        candidate.inputs.push(base.requirement);
        candidate.epoch = candidate.epoch.max(base.epoch);
        let under = builder.under(request.condition);
        // premises: CONST L and CONST U under g; a constant endpoint needs no
        // wire, and a lower endpoint of 0 needs no cut at all.
        let mut cuts = 0i128;
        let mut ports: Vec<TermId> = Vec::new();
        for (term, is_lower) in [(lower, true), (upper, false)] {
            match builder.ask_for(&mut candidate, Question::Const(term), request.epoch, under) {
                Verdict::Constant(value) if is_lower && value == 0 => continue,
                Verdict::Constant(_) => cuts += 1,
                _ => {
                    cuts += 1;
                    ports.push(term);
                }
            }
        }
        for term in ports {
            let row = builder.child(request, Target::Term(term), Family::Wire, request.condition)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        let gated = !builder.terms.is_true(request.condition);
        if gated {
            let row = builder.child(
                request,
                Target::BoolTerm(request.condition),
                Family::Wire,
                builder.terms.tt(),
            )?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        let step = LineageStep::Slice { lo: lower, hi: upper, step: 1, axis: 0 };
        let Some(mut candidate) =
            self.finish(builder, request, key, array, candidate, lambda, step, None)?
        else {
            return Ok(Vec::new());
        };
        // emitter: one cut vector per present endpoint — a thermometer over a
        // dense base, a Boolean prefix over a ranked one — and one `and` per
        // slot per conjunct beyond the first.
        let slots = lambda.layout.slots();
        let dense = lambda.layout.is_dense();
        for _ in 0..cuts {
            candidate = if dense {
                candidate.emit(Primitive::Therm { capacity: slots }, 1)
            } else {
                candidate
                    .emit(Primitive::Bool, slots)
                    .emit(Primitive::Mul, 2 * slots - 1)
            };
        }
        let conjuncts = cuts + i128::from(!dense);
        candidate = candidate
            .emit(Primitive::And, (conjuncts - 1).max(0) * slots)
            .emit(Primitive::Mul, 2 * i128::from(gated));
        Ok(vec![candidate])
    }

    /// The `len` port and the output bundle every variant shares.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        array: &crate::phase0::desc::ArrayDesc,
        mut candidate: Candidate,
        lambda: &Layout,
        step: LineageStep,
        ranks: Option<ValueKey>,
    ) -> Result<Option<Candidate>, Reject> {
        // `len`: ℓ for the rank-1 bundle, shape[0] for the rank-≥2 one. A
        // constant extent is substituted and needs no wire.
        if builder.terms.constant_value(array.dims[0]).is_none() {
            let row = builder.child(
                request,
                Target::Term(array.dims[0]),
                Family::Wire,
                request.condition,
            )?;
            let Some(first) = row.entries.first() else {
                return Ok(None);
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        let slots = lambda.layout.slots();
        let mut lineage = lambda.lineage.clone();
        lineage.push(step);
        let out = Layout {
            layout: LayoutKind::Ranked { capacity: slots, ranks },
            root: lambda.root.clone(),
            lineage,
        };
        let interface = if array.rank() == 1 {
            Interface::Masked { dtype: array.dtype, capacity: slots, lambda: out }
        } else {
            Interface::Store(StoreIface {
                owner: key.clone(),
                dtype: array.dtype,
                capacity: slots,
                lambda: out,
            })
        };
        Ok(Some(
            candidate
                .output(request.target.clone(), interface, request.condition)
                .parameter("B", slots)
                .parameter("w", bits_of_capacity(slots) as i128),
        ))
    }
}
