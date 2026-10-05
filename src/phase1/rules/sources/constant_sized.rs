//! `CONSTANT_SIZED` (`revision/phase1/rules/sources/constant_sized.md`).
//!
//! Implements `E-ValueSized` for the constructors of `CONSTANT`'s table when
//! the extent argument is a scalar of origin data and the source declares a
//! maximum M: phase 0 introduced a fresh variable ℓ with `var_range`
//! `0 ≤ ℓ ≤ M`, the declaration `ℓ = max(0, t_n)` and the check `t_n ≤ M`.
//! The rule takes ℓ as a wire through the term requirement for the right-hand
//! side of the declaration, takes the capacity from `MAX ℓ` — which the
//! `var_range` bounds by M — and builds the bundles of `CONSTANT` with
//! T = {0}: literal elements and a thermometer for validity.
//!
//! The `tight` variant is the later-epoch node of §6.2: after
//! `assert len(z) <= 8`, MAX ℓ over Φ_{e′} is 8 instead of M, and
//! [`Builder::capacity_box`] offers both.

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::phi::VarOrigin;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Capacity, Family, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ConstantSized;

impl Rule for ConstantSized {
    fn id(&self) -> RuleId {
        RuleId::ConstantSized
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
        let Some((dtype, dims, length)) = self.matches(builder, &key) else {
            return Ok(Vec::new());
        };

        let mut probe = Candidate::new(RuleId::ConstantSized, None, request.epoch);
        let under = builder.under(request.condition);
        // resource_relations: B = MAX ℓ over Φ_e ∧ g, with the later-epoch
        // variant of §6.2 where a check tightened the length.
        let boxes = builder.capacity_box(&mut probe, &dims, request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);

        let mut out = Vec::new();
        for capacities in boxes {
            let mut candidate = Candidate::new(
                RuleId::ConstantSized,
                Some(if capacities.epoch > request.epoch { "tight" } else { "box" }),
                capacities.epoch,
            );
            candidate.answers = probe.answers.clone();
            candidate.matched.push(key.clone());
            // `len`: the right-hand side of the declaration, whose wire
            // `TERM_TO_WIRES` compiles from the wire of n. The port is not
            // `Represent(ℓ, Wire[Idx])`: that requirement resolves to
            // `LEN_SHAPE` over a bundle of v, which this node produces.
            let row = builder.child(request, Target::Term(length), Family::Wire, request.condition)?;
            let Some(first) = row.entries.first() else {
                continue;
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
            if gated {
                let row = builder.child(
                    request,
                    Target::BoolTerm(request.condition),
                    Family::Wire,
                    builder.terms.tt(),
                )?;
                if let Some(first) = row.entries.first() {
                    candidate.inputs.push(first.requirement);
                    candidate.epoch = candidate.epoch.max(first.epoch);
                }
            }

            let capacity: Capacity = capacities.slots();
            // emitter (X1): `u = therm(ℓ′ over B)`; the elements are literals,
            // so X3 costs nothing where `val = 0`.
            candidate = candidate
                .emit(Primitive::Therm { capacity: capacities.axes[0] }, 1)
                .emit(Primitive::Mul, i128::from(gated))
                .output(
                    request.target.clone(),
                    Interface::Store(StoreIface {
                        owner: key.clone(),
                        dtype,
                        capacity,
                        lambda: Layout::dense(
                            LayoutKind::Box(capacities.axes.clone()),
                            key.clone(),
                        ),
                    }),
                    request.condition,
                )
                .parameter("B", capacity)
                .parameter("w", bits_of_capacity(capacity) as i128);
            // §3.4: one node answers the `Store`, the rank-1 `Masked` and the
            // canonical `Packed`, whose tail the thermometer already zeroes.
            if capacities.axes.len() == 1 {
                candidate = candidate.output(
                    request.target.clone(),
                    Interface::Masked {
                        dtype,
                        capacity,
                        lambda: Layout::dense(
                            LayoutKind::Ranked { capacity, ranks: None },
                            key.clone(),
                        ),
                    },
                    request.condition,
                );
            }
            candidate = candidate.output(
                request.target.clone(),
                Interface::Packed {
                    dtype,
                    capacity,
                    lambda: Layout::dense(LayoutKind::Compact(capacity), key.clone()),
                    canon: true,
                },
                request.condition,
            );
            out.push(candidate);
        }
        Ok(out)
    }
}

impl ConstantSized {
    /// `source_pattern`: the producer names `E-ValueSized` beside a
    /// constructor, and axis 0's dimension term is the variable ℓ phase 0
    /// introduced, whose declaration gives the term the `len` port reads.
    fn matches(
        &self,
        builder: &Builder<'_>,
        key: &ValueKey,
    ) -> Option<(Dtype, Vec<TermId>, TermId)> {
        let entry = builder.entry(key)?;
        let Producer::Applied { rules, .. } = &entry.producer else {
            return None;
        };
        // The producer names the constructor. `E-ValueSized` is a helper
        // phase 0 applies inside it and does not record in the rule list
        // (`../../phase0/rules/constructors.rs`, `e_value_sized`), so what
        // marks a value-sized construct here is the descriptor: axis 0's
        // extent is the variable ℓ that helper introduced.
        if !rules.iter().any(|rule| {
            matches!(
                rule,
                Phase0Rule::EZeros
                    | Phase0Rule::EOnes
                    | Phase0Rule::EArange
                    | Phase0Rule::EValueSized
            )
        }) {
            return None;
        }
        let array = entry.desc.as_array()?;
        if array.rank() == 0 || !super::has_encoding(array.dtype) {
            return None;
        }
        let variable = builder.terms.var_of(array.dims[0])?;
        // The declaration `ℓ = max(0, t_n)` is what makes MAX answerable; its
        // right-hand side is the term the `len` port asks for.
        let info = builder.result.phi.var_info(variable);
        if !matches!(info.origin, VarOrigin::ValueSized { .. }) {
            return None;
        }
        let length = self.declared_length(builder, variable)?;
        Some((array.dtype, array.dims.clone(), length))
    }

    /// The right-hand side of the declaration `ℓ = max(0, t_n)` phase 0
    /// recorded for the variable.
    fn declared_length(&self, builder: &Builder<'_>, variable: crate::phase0::term::VarId) -> Option<TermId> {
        use crate::phase0::term::NodeView;
        for constraint in builder.result.phi.constraints() {
            if !matches!(constraint.kind, crate::phase0::phi::ConstraintKind::Declaration) {
                continue;
            }
            // The declaration is `φ ⇒ (ℓ = t)`; the equality names the
            // variable on one side and the term on the other.
            let Some((left, right)) = self.equality_of(builder, constraint.formula) else {
                continue;
            };
            for (a, b) in [(left, right), (right, left)] {
                if matches!(builder.terms.node_at(a as usize), NodeView::Var(v) if v == variable)
                {
                    return builder.terms.int_at(b);
                }
            }
        }
        None
    }

    /// The equality at the top of a declaration's formula, through the
    /// implication phase 0 guards it with.
    fn equality_of(
        &self,
        builder: &Builder<'_>,
        formula: crate::phase0::term::BoolId,
    ) -> Option<(u32, u32)> {
        use crate::phase0::term::NodeView;
        let mut stack = vec![formula.index() as u32];
        while let Some(at) = stack.pop() {
            match builder.terms.node_at(at as usize) {
                NodeView::Eq(left, right) => return Some((left, right)),
                NodeView::Or(a, b) | NodeView::And(a, b) => stack.extend([a, b]),
                NodeView::Not(inner) => stack.push(inner),
                _ => {}
            }
        }
        None
    }
}
