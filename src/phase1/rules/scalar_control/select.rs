//! `SELECT` (`revision/phase1/rules/scalar_control/select.md`).
//!
//! Realizes a merge: a value phase 0 recorded as the `ite` of two sides on a
//! Boolean `b`. For a scalar it is one `select` on the sides' wires; for an
//! array a slot-wise select between two bundles of the sides in one layout
//! at the larger capacity; for a consumer that reads a batch a select per
//! entry on the two sides' fetched results. `/index` — a composite indexed
//! by a runtime scalar — is not written.
//!
//! A merged *element of a composite* is realized here too. Δ keys the
//! composite and not its elements (`../../phase0/definition.md` §3.3), so
//! such an element is named by its own term and the variable phase 0 minted
//! for it records the element path down to it (§3.4). The rule reads that
//! path, finds the same element in each arm composite, and raises one port
//! per arm element, exactly as for a scalar merge.

use crate::phase0::desc::{ScalarTerm, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::{BoolId, NodeView};

use crate::phase1::build::Builder;
use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::graph::Target;
use crate::phase1::interface::{Capacity, Family, Interface, Layout, LayoutKind};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::{Condition, Question};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Select;

/// One side of a merge: the arm value whose element is read, and the row
/// that element is named by.
struct Side {
    /// The arm's key in Δ, `None` for a literal arm, which is wired into
    /// the select rather than read from a row.
    arm: Option<ValueKey>,
    /// The row the arm's element is named by, `None` when the element
    /// carries no term phase 1 can name.
    target: Option<Target>,
}

impl Rule for Select {
    fn id(&self) -> RuleId {
        RuleId::Select
    }

    fn answers(&self) -> &'static [Family] {
        &[
            Family::Wire,
            Family::Store,
            Family::Masked,
            Family::Packed,
            Family::Values,
        ]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        // source_pattern: the merge value itself, or an element of a merged
        // composite, named by the term the element carries.
        let Some((key, path)) = self.matches(builder, request) else {
            return Ok(Vec::new());
        };
        if request.family != Family::Wire {
            return self.merged_array(builder, request, &key, &path);
        };
        let Some(entry) = builder.entry(&key) else {
            return Ok(Vec::new());
        };
        let Producer::Applied { rules, operands, condition: Some(condition) } = &entry.producer
        else {
            return Ok(Vec::new());
        };
        if !rules.iter().any(|rule| {
            matches!(
                rule,
                Phase0Rule::MergeScalar | Phase0Rule::MergeArray | Phase0Rule::SIf
                    | Phase0Rule::ECondExp
            )
        }) {
            return Ok(Vec::new());
        }
        let operands = operands.clone();
        let condition = *condition;
        let Some(merged) = builder.element_of(&key, &path).and_then(|d| d.as_scalar()) else {
            return Ok(Vec::new());
        };
        // P2: phase 0 rejects a merge whose sides differ in rank or dtype
        // family, so the sides agree here.
        let dtype = merged.dtype;
        let term = merged.term;
        if !super::super::sources::has_encoding(dtype) {
            return Ok(Vec::new());
        }
        // ψ of the merged value: the condition the merge point runs under,
        // which is what P1 is asked under and what the arms are read within.
        let psi = builder.element_psi(&key, &path);
        let sides = self.sides(builder, term, condition, &operands, &path);

        // P1 (variant): CONST b under ψ_v decides the merge, and the node is
        // the selected side with no gate.
        let mut probe = Candidate::new(RuleId::Select, Some("/static"), request.epoch);
        let under_psi = builder.under(psi);
        let decided = builder.ask_for(
            &mut probe,
            Question::ConstBool(condition),
            request.epoch,
            under_psi,
        );
        if let Verdict::ConstantBool(taken) = decided {
            let side = if taken { sides.first() } else { sides.last() };
            let Some(side) = side else {
                return Ok(Vec::new());
            };
            let mut candidate = probe.clone().output(
                request.target.clone(),
                Interface::Wire(dtype),
                request.condition,
            );
            candidate.matched.push(key.clone());
            // A literal side needs no port; the constant is the answer.
            if let (Some(arm), Some(target)) = (side.arm.clone(), side.target.clone()) {
                // The decided arm is taken wherever the merge runs, so it is
                // read under ψ itself.
                let port =
                    self.port(builder, &mut candidate, request, &arm, target, &path, under_psi)?;
                if !port {
                    return Ok(Vec::new());
                }
            }
            return Ok(vec![candidate]);
        }

        // `/scalar`: the condition's wire and one port per side.
        let mut candidate = Candidate::new(RuleId::Select, Some("/scalar"), request.epoch);
        candidate.matched.push(key.clone());
        candidate.answers = probe.answers.clone();
        // `cond` is raised under ψ_b, the Boolean's own existence condition
        // (`select.md`, `required_input_ports`), and not under `true`: the
        // condition of a merge that a `return` inside a branch produced only
        // exists on the path that reached it, and a row under `true` has no
        // provider there.
        let cond_psi = builder.target_psi(&Target::BoolTerm(condition), builder.terms.tt());
        let cond_under =
            builder.child_under_g(&mut candidate, request, cond_psi, "the merge condition")?;
        let cond_row =
            builder.child(request, Target::BoolTerm(condition), Family::Wire, cond_under)?;
        let Some(cond_entry) = cond_row.entries.first() else {
            return Ok(Vec::new());
        };
        candidate.inputs.push(cond_entry.requirement);
        candidate.epoch = candidate.epoch.max(cond_entry.epoch);
        for (at, side) in sides.iter().enumerate() {
            let (Some(arm), Some(target)) = (side.arm.clone(), side.target.clone()) else {
                continue;
            };
            let arm_psi = self.arm_psi(builder, &arm, &path, &target);
            let under = self.arm_condition(builder, psi, condition, arm_psi, at, sides.len());
            let port =
                self.port(builder, &mut candidate, request, &arm, target, &path, under)?;
            if !port {
                return Ok(Vec::new());
            }
        }
        // emitter: `z = select(b̃, x_1, x_2)`, one `select` per merge, on the
        // backend's float instruction when the merged dtype is Float. A fold
        // of several returns is a chain of them.
        let gate = if super::super::sources::is_float(dtype) {
            Primitive::Float(FloatOp::Select)
        } else {
            Primitive::Select
        };
        candidate = candidate.emit(gate, (sides.len() as i128 - 1).max(1));
        candidate = candidate.output(
            request.target.clone(),
            Interface::Wire(dtype),
            request.condition,
        );
        Ok(vec![candidate])
    }
}

impl Select {
    /// The `/array`, `/values` and array `/static` variants of `select.md`:
    /// a merged *array*.
    ///
    /// `/array` takes one bundle of each side in the row's own family and
    /// selects slot by slot — values, validity, keys, ℓ and the runtime
    /// shape wires — at the larger capacity. `/values` answers a `Fetch` by
    /// selecting the two sides' fetched results and flags per entry. Under
    /// CONST b the node is the selected side's bundle and costs nothing.
    fn merged_array(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        key: &ValueKey,
        path: &[usize],
    ) -> Result<Vec<Candidate>, Reject> {
        // Only the value itself is a merged array; an element of a merged
        // composite is a scalar, which the `Wire` row above answers.
        if !path.is_empty() {
            return Ok(Vec::new());
        }
        let Some(entry) = builder.entry(key) else {
            return Ok(Vec::new());
        };
        let Producer::Applied { rules, operands, condition: Some(condition) } = &entry.producer
        else {
            return Ok(Vec::new());
        };
        if !rules.iter().any(|rule| {
            matches!(
                rule,
                Phase0Rule::MergeScalar | Phase0Rule::MergeArray | Phase0Rule::SIf
                    | Phase0Rule::ECondExp
            )
        }) {
            return Ok(Vec::new());
        }
        let operands = operands.clone();
        let condition = *condition;
        let Some(array) = builder.array(key).cloned() else {
            return Ok(Vec::new());
        };
        // P2: the sides agree in rank and dtype — phase 0 rejects a merge
        // that does not — and τ has a wire encoding.
        if array.rank() == 0 || !super::super::sources::has_encoding(array.dtype) {
            return Ok(Vec::new());
        }
        let sides = self.sides(builder, ScalarTerm::Opaque, condition, &operands, path);
        let [left, right] = sides.as_slice() else {
            return Ok(Vec::new());
        };
        let (Some(left_arm), Some(right_arm)) = (left.arm.clone(), right.arm.clone()) else {
            return Ok(Vec::new());
        };
        let psi = builder.element_psi(key, path);

        // P1: CONST b under ψ_v decides the merge, and the node is the taken
        // side's bundle with no gate.
        let mut probe = Candidate::new(RuleId::Select, Some("/static"), request.epoch);
        let under_psi = builder.under(psi);
        let decided = builder.ask_for(
            &mut probe,
            Question::ConstBool(condition),
            request.epoch,
            under_psi,
        );
        if let Verdict::ConstantBool(taken) = decided {
            let arm = if taken { left_arm } else { right_arm };
            let row = self.arm_row(builder, request, &mut probe, &arm, under_psi)?;
            let mut out = Vec::new();
            for port in &row.entries {
                let mut candidate = probe.clone();
                candidate.matched.push(key.clone());
                candidate.inputs.push(port.requirement);
                candidate.epoch = candidate.epoch.max(port.epoch);
                out.push(candidate.output(
                    request.target.clone(),
                    port.interface.clone(),
                    request.condition,
                ));
            }
            return Ok(out);
        }

        // `cond`: the Boolean's wire, shared by every pair below.
        let mut base = Candidate::new(RuleId::Select, None, request.epoch);
        base.matched.push(key.clone());
        base.answers = probe.answers.clone();
        // Under ψ_b, as in the `/scalar` form above (`select.md`).
        let cond_psi = builder.target_psi(&Target::BoolTerm(condition), builder.terms.tt());
        let cond_under =
            builder.child_under_g(&mut base, request, cond_psi, "the merge condition")?;
        let cond_row =
            builder.child(request, Target::BoolTerm(condition), Family::Wire, cond_under)?;
        let Some(cond_entry) = cond_row.entries.first() else {
            return Ok(Vec::new());
        };
        base.inputs.push(cond_entry.requirement);
        base.epoch = base.epoch.max(cond_entry.epoch);

        // Each side is read under the condition its arm is taken on.
        let left_under = builder.under_arm(psi, condition, false);
        let right_under = builder.under_arm(psi, condition, true);
        let left_row = self.arm_row(builder, request, &mut base, &left_arm, left_under)?;
        let right_row = self.arm_row(builder, request, &mut base, &right_arm, right_under)?;
        if request.family == Family::Values {
            let (Some(left_port), Some(right_port)) =
                (left_row.entries.first(), right_row.entries.first())
            else {
                return Ok(Vec::new());
            };
            let Interface::Values { queries, .. } = left_port.interface else {
                return Ok(Vec::new());
            };
            let mut candidate = base;
            candidate.variant = Some("/values");
            for port in [left_port, right_port] {
                candidate.inputs.push(port.requirement);
                candidate.epoch = candidate.epoch.max(port.epoch);
            }
            // emitter: `y_q` and `act_q` per entry.
            candidate = candidate.emit(
                self.value_gate(array.dtype),
                queries,
            );
            candidate = candidate.emit(Primitive::Select, queries);
            let Some(batch) = request.batch.clone() else {
                return Ok(Vec::new());
            };
            return Ok(vec![candidate.answers_fetch(
                request.target.clone(),
                batch,
                Interface::Values { dtype: array.dtype, queries },
                request.condition,
            )]);
        }

        // `/array`: one node per pair of bundles of one layout kind, at the
        // larger capacity.
        let under = builder.under(request.condition);
        let mut runtime = 0i128;
        for &dimension in &array.dims {
            if !matches!(
                builder.ask_for(&mut base, Question::Const(dimension), request.epoch, under),
                Verdict::Constant(_)
            ) {
                runtime += 1;
            }
        }
        let mut out = Vec::new();
        for left_port in &left_row.entries {
            for right_port in &right_row.entries {
                let (Some(left_lambda), Some(right_lambda)) =
                    (left_port.interface.lambda(), right_port.interface.lambda())
                else {
                    continue;
                };
                let Some(layout) = merged_layout(&left_lambda.layout, &right_lambda.layout)
                else {
                    continue;
                };
                let slots = layout.slots();
                let mut candidate = base.clone();
                candidate.variant = Some("/array");
                for port in [left_port, right_port] {
                    candidate.inputs.push(port.requirement);
                    candidate.epoch = candidate.epoch.max(port.epoch);
                }
                // emitter: the value selects, then the wires the layout
                // carries beside them.
                candidate = candidate.emit(self.value_gate(array.dtype), slots);
                let (companions, lengths) = match &layout {
                    // box and ranked select the validity too; keyed the keys;
                    // compact and ranked the one length wire.
                    LayoutKind::Box(_) => (slots, 0),
                    LayoutKind::Ranked { .. } => (slots, 1),
                    LayoutKind::Keyed { .. } => (slots, 0),
                    LayoutKind::Compact(_) => (0, 1),
                };
                candidate = candidate.emit(
                    Primitive::Select,
                    companions + lengths + if matches!(layout, LayoutKind::Box(_)) { runtime } else { 0 },
                );
                let canon = matches!(
                    (&left_port.interface, &right_port.interface),
                    (
                        Interface::Packed { canon: true, .. },
                        Interface::Packed { canon: true, .. }
                    )
                );
                let lambda = Layout::dense(layout, key.clone());
                let interface = match request.family {
                    Family::Store => Interface::Store(crate::phase1::interface::StoreIface {
                        owner: key.clone(),
                        dtype: array.dtype,
                        capacity: slots,
                        lambda,
                    }),
                    Family::Masked => Interface::Masked {
                        dtype: array.dtype,
                        capacity: slots,
                        lambda,
                    },
                    Family::Packed => Interface::Packed {
                        dtype: array.dtype,
                        capacity: slots,
                        lambda,
                        canon,
                    },
                    _ => continue,
                };
                candidate = candidate
                    .output(request.target.clone(), interface, request.condition)
                    .parameter("B", slots);
                out.push(candidate);
            }
        }
        Ok(out)
    }

    /// The row of one arm of an array merge, in the family of the row being
    /// filled, raised under the arm's own ψ with §4's premise checked
    /// against the condition the arm is taken on.
    fn arm_row(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        candidate: &mut Candidate,
        arm: &ValueKey,
        under: Condition,
    ) -> Result<crate::phase1::table::Row, Reject> {
        let psi = builder.psi(arm);
        let named = arm.encode();
        let condition = builder.child_condition(candidate, under, psi, &named)?;
        match request.family {
            Family::Values => {
                let Some(batch) = request.batch.clone() else {
                    return Ok(crate::phase1::table::Row::default());
                };
                builder.fetch(request, arm, &batch, condition)
            }
            family => builder.child(request, Target::Value(arm.clone()), family, condition),
        }
    }

    /// The gate one element select is: `select_f` on a Float element, the
    /// field `select` otherwise (`../README.md` §5).
    fn value_gate(&self, dtype: crate::phase0::desc::Dtype) -> Primitive {
        if super::super::sources::is_float(dtype) {
            Primitive::Float(FloatOp::Select)
        } else {
            Primitive::Select
        }
    }

    /// `source_pattern`: the value whose merge this row is about, and the
    /// element path inside it the row names.
    ///
    /// A `Represent(v, Wire)` row is the value itself at the empty path. A
    /// term row is an element of a merged composite when the term is a
    /// variable phase 0 introduced there (`rules/README.md` §2).
    fn matches(
        &self,
        builder: &Builder<'_>,
        request: &RowRequest,
    ) -> Option<(ValueKey, Vec<usize>)> {
        let index = match &request.target {
            // An element of a merged composite has a key of its own
            // (`../../../phase0/definition.md` §3.3). The merge it is an
            // element of is the composite's, so the row is that merge read
            // at the element's path — which is what a merged element whose
            // descriptor carries no term is named by.
            Target::Value(key) => {
                return match builder.element_of_key(key) {
                    Some((composite, at)) => Some((composite, vec![at])),
                    None => Some((key.clone(), Vec::new())),
                }
            }
            Target::Term(t) => t.index() as u32,
            Target::BoolTerm(b) => b.index() as u32,
        };
        let (key, path) = builder.variable_origin(index)?;
        if path.is_empty() {
            return None;
        }
        // The path must lead back to this very term, or the row is about
        // something else that happens to mention the variable.
        if builder.element_target(&key, &path)? != request.target {
            return None;
        }
        Some((key, path))
    }

    /// One side's port: the row of the arm's element, raised under that
    /// element's own ψ, with §4's premise checked against `under`.
    ///
    /// Returns whether the row had a provider; a side with none drops the
    /// candidate.
    #[allow(clippy::too_many_arguments)]
    fn port(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        request: &RowRequest,
        arm: &ValueKey,
        target: Target,
        path: &[usize],
        under: Condition,
    ) -> Result<bool, Reject> {
        let psi = self.arm_psi(builder, arm, path, &target);
        let named = target.render();
        let condition = builder.child_condition(candidate, under, psi, &named)?;
        let row = builder.child(request, target, Family::Wire, condition)?;
        let Some(first) = row.entries.first() else {
            return Ok(false);
        };
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
        Ok(true)
    }

    /// ψ of the element of an arm this merge reads: the arm's element sits
    /// at the same path inside the arm as the merged element does inside the
    /// merge. An arm named by the value that computes the element's term
    /// carries its own ψ instead. An element named by a term that some value
    /// of Δ owns is that value's row, so its ψ is the owner's (§3.2).
    fn arm_psi(
        &self,
        builder: &Builder<'_>,
        arm: &ValueKey,
        path: &[usize],
        target: &Target,
    ) -> BoolId {
        let own = match builder.element_of(arm, path) {
            Some(_) => builder.element_psi(arm, path),
            None => builder.element_psi(arm, &[]),
        };
        builder.target_psi(target, own)
    }

    /// The condition one arm is read under (`select.md`,
    /// `correctness_theorem`): the built term `ψ_v ∧ b` for the first side of
    /// a two-way merge and `ψ_v ∧ ¬b` for the second.
    ///
    /// A fold of several returns is a chain of two-way merges of which Δ
    /// records only the outermost condition, so the inner arms have no
    /// condition to add; what phase 0 recorded as the path each return was
    /// taken on is that arm's ψ, and that is the condition it is read under.
    fn arm_condition(
        &self,
        builder: &mut Builder<'_>,
        psi: BoolId,
        condition: BoolId,
        arm_psi: BoolId,
        at: usize,
        sides: usize,
    ) -> Condition {
        if sides != 2 {
            return builder.under_arm(psi, arm_psi, false);
        }
        builder.under_arm(psi, condition, at != 0)
    }

    /// The sides of the merge, the `b`-true one first.
    ///
    /// Phase 0's two merge sites record their operands in opposite orders —
    /// a scope merge as (then, else), a fold of several `return`s with the
    /// last return first in the `ite` — so the order is read off the merged
    /// term where the merge is shape-origin and the term is the `ite`
    /// itself. The operand list is not always the answer either: an element
    /// of an unpacked tuple is named by the composite's key
    /// (`implementation.md` §5a), so the operand is the composite and the
    /// side is the scalar inside it; the arm's term names that scalar
    /// through its owning value (`rules/README.md` §3.2).
    ///
    /// A data merge carries a fresh variable instead, and its declaration
    /// `φ ⇒ v = ite(b, t_1, t_2)` is a constraint of Φ rather than the
    /// descriptor's term; the operands are then taken in Δ's order, which
    /// names the same two values and so gives the same node with the two
    /// ports exchanged.
    fn sides(
        &self,
        builder: &Builder<'_>,
        term: ScalarTerm,
        condition: BoolId,
        operands: &[ValueKey],
        path: &[usize],
    ) -> Vec<Side> {
        let ordered = || -> Option<Vec<Side>> {
            let ScalarTerm::Int(merged) = term else {
                return None;
            };
            let NodeView::Ite(c, a, b) = builder.terms.node_at(merged.index()) else {
                return None;
            };
            if c != condition.index() as u32 {
                return None;
            }
            Some(vec![
                self.side_of(builder, operands, path, a)?,
                self.side_of(builder, operands, path, b)?,
            ])
        };
        ordered().unwrap_or_else(|| {
            let mut sides: Vec<Side> = operands
                .iter()
                .map(|arm| Side {
                    arm: Some(arm.clone()),
                    target: builder.element_target(arm, path),
                })
                .collect();
            // The fold of several `return`s puts the *last* return first in
            // the `ite` and records that return's path condition as the
            // merge's, while Δ lists the operands in source order
            // (`../../../phase0/implementation.md` §5a). An element that
            // carries no term — a `Float` — has no `ite` to read the order
            // off, so it is read off the sides' own execution conditions:
            // the side whose ψ is the merge's condition is the `b`-true one.
            if sides.len() == 2 {
                let psi_of = |side: &Side| {
                    let (Some(arm), Some(target)) = (side.arm.as_ref(), side.target.as_ref())
                    else {
                        return None;
                    };
                    Some(self.arm_psi(builder, arm, path, target))
                };
                let (first, second) = (psi_of(&sides[0]), psi_of(&sides[1]));
                if second == Some(condition) && first != Some(condition) {
                    sides.swap(0, 1);
                }
            }
            sides
        })
    }

    /// The side one arm of the merged `ite` names: the operand whose element
    /// at `path` carries that term, or the value that computes it. A literal
    /// arm has no row and is wired into the select.
    fn side_of(
        &self,
        builder: &Builder<'_>,
        operands: &[ValueKey],
        path: &[usize],
        index: u32,
    ) -> Option<Side> {
        if let Some(term) = builder.terms.int_at(index) {
            if builder.terms.constant_value(term).is_some() {
                return Some(Side { arm: None, target: None });
            }
        }
        let carries = |key: &ValueKey| {
            builder
                .element_of(key, path)
                .and_then(|desc| desc.as_scalar())
                .map(|scalar| match scalar.term {
                    ScalarTerm::Int(t) => t.index() as u32 == index,
                    ScalarTerm::Bool(t) => t.index() as u32 == index,
                    ScalarTerm::Opaque => false,
                })
                .unwrap_or(false)
        };
        if let Some(arm) = operands.iter().find(|key| carries(key)) {
            return Some(Side {
                arm: Some(arm.clone()),
                target: builder.element_target(arm, path),
            });
        }
        // The arm is not an operand of this merge: the value that owns the
        // hash-consed term is the one to read.
        let owner = builder.owner_of_term(index)?.clone();
        Some(Side {
            target: Some(Target::Value(owner.clone())),
            arm: Some(owner),
        })
    }
}

/// The layout of a merged bundle: the sides' common layout kind at the
/// larger capacity (`select.md`, `produced_output_bundle`). Two sides of
/// different kinds are not merged — the slot of one is not the slot of the
/// other — and a `ranked` merge keeps no `Ranks` node, since the merged
/// validity is neither side's mask.
fn merged_layout(left: &LayoutKind, right: &LayoutKind) -> Option<LayoutKind> {
    Some(match (left, right) {
        (LayoutKind::Box(a), LayoutKind::Box(b)) if a.len() == b.len() => LayoutKind::Box(
            a.iter().zip(b).map(|(x, y)| (*x).max(*y)).collect::<Vec<Capacity>>(),
        ),
        (LayoutKind::Compact(a), LayoutKind::Compact(b)) => LayoutKind::Compact((*a).max(*b)),
        (LayoutKind::Ranked { capacity: a, .. }, LayoutKind::Ranked { capacity: b, .. }) => {
            LayoutKind::Ranked { capacity: (*a).max(*b), ranks: None }
        }
        (LayoutKind::Keyed { capacity: a }, LayoutKind::Keyed { capacity: b }) => {
            LayoutKind::Keyed { capacity: (*a).max(*b) }
        }
        _ => return None,
    })
}
