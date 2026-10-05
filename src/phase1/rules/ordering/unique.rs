//! `UNIQUE` (`revision/phase1/rules/ordering/unique.md`).
//!
//! `u = unique(a)`: the distinct elements of the flattened operand in
//! ascending order, of length `c′`. The sort kernel of `sort.md` on the
//! operand's (value, validity) multiset gives a sorted copy y; a flag `f_j`
//! marks the first slot of every run of equal values; `(y, f, Σf)` is
//! already a `Masked` of u (variant `masked`). A `Packed` of capacity
//! `B′ = MAX c′` is obtained by a second `perm` that routes the flagged
//! slots to a prefix (variant `packed`). With `return_inverse`, a third
//! `perm` carries each operand element's run index back to its own slot.
//!
//! **Where the two values are.** `np.unique(a, return_inverse=True)` gives
//! Δ one Composite key whose elements are u and w; the program's
//! `vals, idx = …` binds each element under a key of its own, as a
//! single-operand `S-Assign` over that Composite. The rule matches either
//! binding, reaches the other through the Composite, and offers one node
//! with both values' bundles among its outputs (§3.4) — so the two rows are
//! filled separately and hold the same node.

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::{bits_of_capacity, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Capacity, Family, Interface, Layout, LayoutKind, StoreIface,
};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Unique;

/// What the source pattern read.
struct Matched {
    /// The operand a.
    operand: ValueKey,
    /// The value that holds the distinct elements.
    unique: ValueKey,
    /// The value that holds the inverse, with `return_inverse`.
    inverse: Option<ValueKey>,
    /// The key whose watermark is the node's default epoch: the call's.
    origin: ValueKey,
}

impl Rule for Unique {
    fn id(&self) -> RuleId {
        RuleId::Unique
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Masked, Family::Packed, Family::Store]
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
        let Some(array) = builder.array(&matched.unique).cloned() else {
            return Ok(Vec::new());
        };
        let Some(order) = super::order_key(builder, array.dtype) else {
            return Ok(Vec::new());
        };
        let width = super::ord_width(order);
        let Some(source) = builder.array(&matched.operand).cloned() else {
            return Ok(Vec::new());
        };
        if source.rank() == 0 || array.rank() == 0 {
            return Ok(Vec::new());
        }

        let mut probe = Candidate::new(RuleId::Unique, None, request.epoch);
        let under = builder.under(request.condition);
        // premises: CONST dims_a[0] (rank 1) or CONST size(a) selects the
        // constant fold; neither answer drops the rule.
        let extent = if source.rank() == 1 {
            source.dims[0]
        } else {
            match self.size(builder, &source.dims) {
                Some(term) => term,
                None => return Ok(Vec::new()),
            }
        };
        let constant = match builder.ask_for(&mut probe, Question::Const(extent), request.epoch, under)
        {
            Verdict::Constant(value) => Some(value.max(0)),
            _ => None,
        };
        // `B′ = MAX c′` over Φ_e ∧ g, with the later-epoch variant of §6.2.
        let packed = builder.capacity_box(&mut probe, &array.dims[..1], request.epoch, under);
        let gated = !builder.terms.is_true(request.condition);

        let mut out = Vec::new();
        for family in [Family::Store, Family::Masked, Family::Packed] {
            let psi = builder.psi(&matched.operand);
            let condition =
                builder.child_under_g(&mut probe, request, psi, &matched.operand.encode())?;
            let row = builder.child(
                request,
                Target::Value(matched.operand.clone()),
                family,
                condition,
            )?;
            for entry in &row.entries.clone() {
                let (Some(operand), Some(lambda)) = (
                    super::operand_of(&entry.interface),
                    entry.interface.lambda().cloned(),
                ) else {
                    continue;
                };
                // Rank ≥ 2 is read as the slots in C-order, which only a box
                // or compact bundle offers (README §2.1).
                if source.rank() > 1 && !operand.dense {
                    continue;
                }
                let capacity = operand.capacity;
                let mut candidate = Candidate::new(RuleId::Unique, Some("masked"), request.epoch);
                candidate.answers = probe.answers.clone();
                candidate.matched.push(matched.unique.clone());
                candidate.epoch = candidate.epoch.max(builder.watermark(&matched.origin));
                candidate.inputs.push(entry.requirement);
                candidate.epoch = candidate.epoch.max(entry.epoch);

                // Step 1, the sort kernel; step 2, the run flags.
                let inverse = u32::from(matched.inverse.is_some());
                if operand.thermometer && constant.is_none() {
                    candidate = candidate.emit(Primitive::Therm { capacity }, 1);
                }
                let pairs = constant.unwrap_or(capacity) - 1;
                let dense = operand.dense && constant.is_some();
                candidate = candidate
                    .emit(
                        Primitive::Perm {
                            capacity: if dense { constant.unwrap_or(capacity) } else { capacity },
                            arity: if dense { 1 } else { 2 } + inverse,
                        },
                        1,
                    )
                    .emit(Primitive::Witness, (1 + inverse as i128) * capacity);
                // The unique values are strictly increasing.
                candidate =
                    super::order_check(candidate, order, pairs, 0, true, constant.is_none());
                // `e_j = eq(ord(y_j), ord(y_{j−1}))` and
                // `f_j = and(m′_j, not(e_j))`, one pair per slot after the
                // first; the `and` folds away where m′ is constant.
                candidate = candidate
                    .emit(Primitive::Eq, pairs)
                    .emit(Primitive::And, if constant.is_none() { pairs } else { 0 });
                if inverse == 1 {
                    // Step 4: the third permutation carries the run index of
                    // every operand element back to its own slot.
                    candidate = candidate
                        .emit(Primitive::Perm { capacity, arity: 3 }, 1)
                        .emit(Primitive::Witness, capacity);
                }
                candidate = candidate
                    .parameter("B", capacity)
                    .parameter("beta", width as i128)
                    .parameter("b", bits_of_capacity(capacity) as i128);
                // The `masked` bundle, and the `inverse` bundle beside it.
                let masked = candidate
                    .clone()
                    .output(
                        Target::Value(matched.unique.clone()),
                        Interface::Masked {
                            dtype: array.dtype,
                            capacity,
                            lambda: Layout::dense(
                                LayoutKind::Ranked { capacity, ranks: None },
                                matched.unique.clone(),
                            ),
                        },
                        request.condition,
                    );
                let masked =
                    self.with_inverse(masked, request, &matched, &lambda, capacity, source.rank());
                out.push(masked.clone());

                // `packed`: the same node plus the pack step.
                for capacities in &packed {
                    let target = capacities.axes[0].min(capacity);
                    let mut node = masked.clone();
                    node.variant = Some("packed");
                    node.epoch = node.epoch.max(capacities.epoch);
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
                        node.inputs.push(first.requirement);
                        node.epoch = node.epoch.max(first.epoch);
                    }
                    node = node
                        .emit(Primitive::Mul, i128::from(gated) * capacity)
                        .emit(Primitive::Therm { capacity: target }, 1)
                        .emit(Primitive::Mul, target)
                        .emit(Primitive::Perm { capacity, arity: 3 }, 1)
                        .emit(Primitive::Witness, 2 * capacity + target)
                        .parameter("B_packed", target);
                    let lambda = Layout::dense(
                        LayoutKind::Compact(target),
                        matched.unique.clone(),
                    );
                    let plain = node.clone().output(
                        Target::Value(matched.unique.clone()),
                        Interface::Packed {
                            dtype: array.dtype,
                            capacity: target,
                            lambda: lambda.clone(),
                            canon: false,
                        },
                        request.condition,
                    );
                    out.push(plain);
                    let mut canonical = node
                        .emit(Primitive::Mul, target)
                        .output(
                            Target::Value(matched.unique.clone()),
                            Interface::Packed {
                                dtype: array.dtype,
                                capacity: target,
                                lambda,
                                canon: true,
                            },
                            request.condition,
                        );
                    canonical.variant = Some("packed.canon");
                    out.push(canonical);
                }
            }
        }
        Ok(out)
    }
}

impl Unique {
    /// `source_pattern`: `Applied([…, E-Unique], [a], _)`, matched either on
    /// the value itself or on a binding of one element of the Tuple the
    /// `return_inverse` form produces.
    fn matches(&self, builder: &Builder<'_>, key: &ValueKey) -> Option<Matched> {
        let entry = builder.entry(key)?;
        // The `return_inverse` form: the binding relabels the Composite.
        if let Some(source) = entry.producer.relabels() {
            let composite = builder.entry(source)?;
            if !composite
                .producer
                .rules()
                .iter()
                .all(|rule| *rule == Phase0Rule::EUnique)
                || composite.producer.rules().is_empty()
            {
                return None;
            }
            let elements = composite.desc.as_composite()?;
            if elements.elems.len() != 2 {
                return None;
            }
            let operand = self.operand(builder, &composite.producer)?;
            let (unique, inverse) = self.bindings(builder, source, elements)?;
            if *key != unique && Some(key) != inverse.as_ref() {
                return None;
            }
            return Some(Matched {
                operand,
                unique,
                inverse,
                origin: source.clone(),
            });
        }
        if !entry.producer.rules().iter().all(|rule| *rule == Phase0Rule::EUnique)
            || entry.producer.rules().is_empty()
        {
            return None;
        }
        entry.desc.as_array()?;
        let operand = self.operand(builder, &entry.producer)?;
        Some(Matched {
            operand,
            unique: key.clone(),
            inverse: None,
            origin: key.clone(),
        })
    }

    /// The array operand of an `E-Unique` producer.
    fn operand(&self, builder: &Builder<'_>, producer: &Producer) -> Option<ValueKey> {
        let Producer::Applied { operands, .. } = producer else {
            return None;
        };
        operands
            .iter()
            .find(|operand| builder.array(operand).is_some())
            .cloned()
    }

    /// The Δ keys the two elements of the Tuple are bound under: the values
    /// whose producer relabels the Composite and whose extents are the
    /// element's.
    fn bindings(
        &self,
        builder: &Builder<'_>,
        composite: &ValueKey,
        elements: &crate::phase0::desc::CompositeDesc,
    ) -> Option<(ValueKey, Option<ValueKey>)> {
        let wanted = |at: usize| -> Option<TermId> {
            elements.elems.get(at)?.as_array().map(|array| array.dims[0])
        };
        let bound = |extent: TermId| -> Option<ValueKey> {
            builder
                .result
                .descriptors
                .iter()
                .find(|entry| {
                    entry.producer.relabels() == Some(composite)
                        && entry
                            .desc
                            .as_array()
                            .map(|array| array.dims.first() == Some(&extent))
                            .unwrap_or(false)
                })
                .map(|entry| entry.key.clone())
        };
        let unique = bound(wanted(0)?)?;
        Some((unique, wanted(1).and_then(bound)))
    }

    /// `size(a)` as a term, for the rank-≥2 operand whose flattening is the
    /// slot order. `None` where the product is not in the term language.
    fn size(&self, builder: &mut Builder<'_>, dims: &[TermId]) -> Option<TermId> {
        let mut constant: Capacity = 1;
        let mut runtime: Option<TermId> = None;
        for &extent in dims {
            match builder.terms.constant_value(extent) {
                Some(value) => constant *= value,
                None if runtime.is_none() => runtime = Some(extent),
                None => return None,
            }
        }
        Some(match runtime {
            Some(term) => builder.terms.mul_const(constant, term),
            None => builder.terms.constant(constant),
        })
    }

    /// The `inverse` bundle: the run index of every operand element, on the
    /// operand's own layout, so that it is aligned with the `in` bundle.
    #[allow(clippy::too_many_arguments)]
    fn with_inverse(
        &self,
        candidate: Candidate,
        request: &RowRequest,
        matched: &Matched,
        lambda: &Layout,
        capacity: Capacity,
        rank: usize,
    ) -> Candidate {
        let Some(inverse) = matched.inverse.clone() else {
            return candidate;
        };
        let interface = match &lambda.layout {
            LayoutKind::Ranked { .. } => Interface::Masked {
                dtype: Dtype::Idx,
                capacity,
                lambda: lambda.clone(),
            },
            LayoutKind::Compact(_) => Interface::Packed {
                dtype: Dtype::Idx,
                capacity,
                lambda: Layout::dense(LayoutKind::Compact(capacity), inverse.clone()),
                canon: false,
            },
            // w is the flattening of a rank-≥2 operand, so its logical key
            // is the C-order index of the operand's slot rather than the
            // slot itself: a keyed layout, whose K_p the emitter computes by
            // `lin` over the operand's shape wires (`unique.md`, `inverse`).
            LayoutKind::Box(_) if rank > 1 => Interface::Store(StoreIface {
                owner: inverse.clone(),
                dtype: Dtype::Idx,
                capacity,
                lambda: Layout::dense(LayoutKind::Keyed { capacity }, inverse.clone()),
            }),
            _ => Interface::Store(StoreIface {
                owner: inverse.clone(),
                dtype: Dtype::Idx,
                capacity,
                lambda: lambda.clone(),
            }),
        };
        candidate.output(Target::Value(inverse), interface, request.condition)
    }
}
