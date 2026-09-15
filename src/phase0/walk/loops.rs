//! Loops: `definition.md` §6.2 and the addition `S-For-Range-Bounded`
//! (`implementation.md` §8).
//!
//! `S-For-Unrolled`, `S-For-Elems` and `S-For-Dyn` are one traversal here,
//! because they differ only in where the per-iteration element comes from:
//! the iterable is evaluated once, turned into a list of element descriptors,
//! and the body is walked once per element. Two of them also guard each
//! iteration, and for the same reason: the number of iterations is not known,
//! only bounded. `S-For-Dyn` unrolls a dynamic array to its declared bound
//! with the guard `j < dims[0]`; `S-For-Range-Bounded` unrolls a `range`
//! whose trip count `T` is a term to the cap `C` that $\Phi$'s ranges give
//! `T`, with the guard `j < T`. A `range` bound that has no usable maximum —
//! an element value, whose only bound is the int64 dtype range — is
//! rejected, as §6.2 says: the baseline silently makes it zero iterations
//! (`coverage.md` §9).
//!
//! `S-While-Bounded` unrolls to a cap: the baseline's structural cap when the
//! loop matches its pattern ([`super::while_cap`]), otherwise `loop_limit`.
//! After the cap,
//! `Φ += ((φ ∧ b_1 ∧ … ∧ b_C) ⇒ false)`: the program is defined only if the
//! loop exits within the cap, which is the baseline's silent stop made
//! explicit.

use crate::ast::{ASTForInStatement, ASTNode, ASTWhileStatement};
use crate::phase0::desc::{ArrayDesc, Desc, ScalarTerm, ValueKey};
use crate::phase0::rules::VarRange;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;
use crate::phase0::Reject;

use super::ranges::{as_range_call, RangePlan};
use super::{Flow, Walker};
use crate::phase0::rules::Rule;

/// Where one iteration's element comes from (§6.2).
///
/// The three arms are the three ways `for x in v` produces `x`. Only the
/// first holds descriptors: the other two *build* the element inside the
/// iteration that binds it, so that its $\psi$ is the path condition of
/// that iteration — the guard included (§2, "a value created in this state
/// gets $\psi = \varphi$").
enum Elements {
    /// `S-For-Elems` over a composite: the element descriptors exist
    /// already, each with the $\psi$ of the expression that built it.
    Ready(Vec<Desc>),
    /// `S-For-Unrolled` and `S-For-Range-Bounded`: the shape scalar
    /// `t_lo + j * step`, which takes no key (`implementation.md` §5a).
    Range { start: TermId, step: i128 },
    /// `S-For-Elems` over an array and `S-For-Dyn`: the rank−1
    /// sub-descriptor of axis 0, under the key §5a rule 4 allocates before
    /// the loop is entered.
    Slots {
        array: ArrayDesc,
        keys: Vec<ValueKey>,
    },
}

/// One element of an iterated or unpacked value: its descriptor and its key
/// in $\Delta$, when it has one.
///
/// An element of an iterated array takes a key of its own (§5a rule 4); an
/// element of a composite is part of the composite's value and is named by
/// the composite's key; an element of a `range` is a shape scalar and takes
/// no key at all (`implementation.md` §5a).
#[derive(Debug, Clone)]
pub(crate) struct Element {
    pub desc: Desc,
    pub key: Option<ValueKey>,
}

impl Element {
    pub fn new(desc: Desc, key: Option<ValueKey>) -> Self {
        Element { desc, key }
    }
}

/// One `for`'s unrolling: the elements, how many iterations, and the guard.
struct ForPlan {
    elements: Elements,
    /// The key of the iterated value, when it has one: the operand every
    /// element's producer names.
    source: Option<ValueKey>,
    count: usize,
    /// `Some(t)` for `S-For-Dyn` and `S-For-Range-Bounded`, whose trip count
    /// is only bounded: iteration `j` runs under `j < t` (§6.2).
    extent: Option<TermId>,
}

impl ForPlan {
    /// The element of iteration `j`, built under the iteration's own path
    /// condition.
    fn element(&self, w: &mut Walker, j: usize) -> Result<Element, Reject> {
        match &self.elements {
            Elements::Ready(values) => values
                .get(j)
                .cloned()
                .map(|desc| Element::new(desc, self.source.clone()))
                .ok_or_else(|| self.no_element(j)),
            // A `range` element is a shape scalar and takes no key of its
            // own (§5a); the rule that builds it belongs to the iterable.
            Elements::Range { start, step } => {
                let (start, step) = (*start, *step);
                let desc = w.in_iterable(self.source.as_ref(), |w| {
                    Ok(w.range_element(start, step, j))
                })?;
                Ok(Element::new(desc, None))
            }
            Elements::Slots { array, keys } => {
                let array = array.clone();
                let key = keys.get(j).cloned().ok_or_else(|| self.no_element(j))?;
                let desc = w.array_element(&array, &key, self.source.as_ref())?;
                Ok(Element::new(desc, Some(key)))
            }
        }
    }

    /// The loop asked for an element the plan does not have, which the
    /// construction of [`ForPlan`] rules out; reported rather than panicked
    /// because nothing here may fail on program-derived data.
    fn no_element(&self, j: usize) -> Reject {
        Reject::Unsupported {
            construct: format!("iteration {} of a {}-element loop", j, self.count),
        }
    }

    /// The compile-time value of each element, for the structural `while`
    /// cap ([`super::while_cap::WhileCap::enter_for`]). Only a `range` of
    /// constants and a composite of constants have one.
    fn constant_values(&self, w: &Walker) -> Vec<Option<i128>> {
        match &self.elements {
            Elements::Ready(values) => values.iter().map(|d| w.constant_of(d)).collect(),
            Elements::Range { start, step } => {
                let lo = w.an.terms.constant_value(*start);
                (0..self.count)
                    .map(|j| lo.map(|lo| lo + step * j as i128))
                    .collect()
            }
            Elements::Slots { .. } => vec![None; self.count],
        }
    }
}

/// `S-For-Unrolled` / `S-For-Elems` / `S-For-Dyn` / `S-For-Range-Bounded`
/// (§6.2, `implementation.md` §8).
pub(super) fn for_in(w: &mut Walker, n: &ASTForInStatement) -> Result<Flow, Reject> {
    let plan = w.for_plan(&n.iter_expr)?;
    w.while_cap.enter_for(&n.target, &plan.constant_values(w));
    w.enter_loop();
    let mark = w.an.keys.enter_loop();
    for index in 0..plan.count {
        w.an.keys.set_iteration(index as u32);
        w.start_iteration();
        // The guard joins `φ` before the element is built, so the element's
        // `ψ` carries it: `definition.md` §6.2 runs iteration `j` under
        // `φ ∧ (j < T)` and §2 gives a value created there `ψ = φ`.
        if let Some(d) = plan.extent {
            let guard = w.below(index, d);
            w.guard_iteration(guard);
        }
        let element = plan.element(w, index)?;
        let bound = w.constant_of(&element.desc);
        w.while_cap.note_element(bound);
        w.assign_target(&n.target, element.desc, element.key.as_ref())?;
        w.block(&n.block)?;
        if w.loop_is_dead() {
            break;
        }
    }
    w.an.keys.exit_loop(mark);
    let completed = w.exit_loop();
    w.while_cap.exit_for();
    w.orelse(&n.orelse, completed)
}

/// `S-While-Bounded` (§6.2).
pub(super) fn while_loop(w: &mut Walker, n: &ASTWhileStatement) -> Result<Flow, Reject> {
    let cap = w.while_cap.cap_for(n).unwrap_or(w.config.loop_limit);
    w.enter_loop();
    let mark = w.an.keys.enter_loop();
    if let Some((guards, guard_key)) = unroll_while(w, n, cap)? {
        // The run is defined only if the loop exits within the cap:
        // `Φ += ((φ ∧ b_1 ∧ … ∧ b_C) ⇒ false)`. The iteration's liveness is
        // reset first, so that the check carries exactly the guards and not
        // whatever a `continue` in the last iteration left behind.
        w.start_iteration();
        w.guard_iteration(guards);
        let ff = w.an.terms.ff();
        // The subject is the guard of the last unrolled iteration: the value
        // whose truth after the cap is what the check forbids
        // (`implementation.md` §5a).
        w.with_subject(&guard_key, |w| {
            w.ctx().check(
                Rule::SWhileBounded.detail("the loop must exit within the cap"),
                ff,
            );
            Ok(())
        })?;
    }
    w.an.keys.exit_loop(mark);
    let completed = w.exit_loop();
    w.orelse(&n.orelse, completed)
}

/// Unroll the body up to `cap` times.
///
/// Iteration `j` runs under `b_1 ∧ … ∧ b_j`, each `b` the guard evaluated in
/// its own iteration. The result is `Some(b_1 ∧ … ∧ b_C, the key of the last
/// guard)` when the cap was reached with that conjunction still satisfiable,
/// which is what the cap check needs, and `None` when the loop provably
/// exited first.
fn unroll_while(
    w: &mut Walker,
    n: &ASTWhileStatement,
    cap: u32,
) -> Result<Option<(crate::phase0::term::BoolId, ValueKey)>, Reject> {
    let mut guards = w.an.terms.tt();
    let mut guard_key = None;
    for iteration in 0..cap {
        w.an.keys.set_iteration(iteration);
        w.start_iteration();
        // The guard expression itself runs only if the previous iterations
        // ran, so its checks are guarded by them too.
        w.guard_iteration(guards);
        let key = w.an.keys.next_expr();
        let test = w.expr_at(&n.test_expr, &key)?;
        let b = w.condition(&test, &key)?;
        guard_key = Some(key);
        if w.an.terms.is_false(b) {
            return Ok(None);
        }
        guards = w.an.terms.and(guards, b);
        w.guard_iteration(b);
        w.block(&n.block)?;
        if w.loop_is_dead() {
            return Ok(None);
        }
    }
    Ok(guard_key.filter(|_| cap > 0).map(|key| (guards, key)))
}

impl Walker<'_> {
    /// How a `for` unrolls: where each element comes from, how many
    /// iterations, and the guard.
    ///
    /// §6.2 states `S-For-Unrolled` over the syntactic form `for i in
    /// range(…)`, and so does `S-For-Range-Bounded`, so a `range` is
    /// recognised here rather than evaluated as a value: the bounded form is
    /// a loop, not a list (see [`super::ranges`]). The keys are the ones the
    /// expression judgment would take — the call node, then its arguments —
    /// so the value-key discipline is unchanged.
    fn for_plan(&mut self, iter_expr: &ASTNode) -> Result<ForPlan, Reject> {
        let Some(call) = as_range_call(self, iter_expr) else {
            let (iterable, source) = self.expr_keyed(iter_expr)?;
            return self.iterable_plan(&iterable, Some(&source));
        };
        self.spend(1)?;
        // The call node's key. A `range` yields shape scalars, which
        // introduce no variable and so need no key of their own
        // (`implementation.md` §5a, "the iterable expression … takes no
        // further keys"), but the node is still a value the walk keys: its
        // entry has no descriptor content, and it is the subject of the
        // questions the loop rule asks about the trip count.
        let key = self.an.keys.next_expr();
        self.record(key.clone(), None, &Desc::None, Producer::Opaque);
        let (plan, _) = self.in_value(&key.clone(), |w| w.range_for_plan(call, key))?;
        Ok(plan)
    }

    /// The unrolling of `for i in range(…)`, inside the value frame of the
    /// `range` call.
    fn range_for_plan(
        &mut self,
        call: &crate::ast::ASTNamedAttribute,
        key: ValueKey,
    ) -> Result<ForPlan, Reject> {
        let args = self.call_args(&call.args, &call.kwargs)?;
        // Unrolling costs at least one unit per iteration, so the iterations
        // are charged before they are built: a `range` the budget cannot
        // afford is a rejection now rather than after it has been
        // materialised.
        let (start, step, count, extent) = match self.range_plan(&args)? {
            RangePlan::Constant { start, step, count } => (start, step, count, None),
            RangePlan::Bounded {
                start,
                step,
                trip,
                cap,
            } => {
                // `S-For-Range-Bounded` unrolls to the cap `Φ`'s ranges give
                // `T`; phase 1 has to answer whether `T` is constant.
                self.ctx()
                    .record_const(Rule::SForRangeBounded.detail("the trip count"), trip);
                (start, step, cap, Some(trip))
            }
        };
        self.spend(count as u64)?;
        Ok(ForPlan {
            elements: Elements::Range { start, step },
            source: Some(key),
            count,
            extent,
        })
    }

    /// The `orelse` block of a `for` or a `while`, run under the condition
    /// that the loop was not left by a `break` — Python's rule. The baseline
    /// always runs a `for ... else` and never runs a `while ... else`
    /// (`coverage.md` §9).
    fn orelse(
        &mut self,
        orelse: &[ASTNode],
        completed: crate::phase0::term::BoolId,
    ) -> Result<Flow, Reject> {
        if orelse.is_empty() {
            return Ok(Flow::Normal);
        }
        self.under_branch(completed, |w| w.block(orelse))?;
        Ok(Flow::Normal)
    }

    /// `index < extent`, the guard of `S-For-Dyn`'s iteration `index`.
    fn below(&mut self, index: usize, extent: TermId) -> crate::phase0::term::BoolId {
        let j = self.an.terms.constant(index as i128);
        self.an.terms.lt(j, extent)
    }

    /// How iterating a value unrolls (§6.2): `S-For-Elems` over a composite
    /// or a static array, `S-For-Dyn` over an array whose length is only
    /// bounded.
    fn iterable_plan(
        &mut self,
        iterable: &Desc,
        source: Option<&ValueKey>,
    ) -> Result<ForPlan, Reject> {
        let iterable = iterable.clone();
        self.in_iterable(source, |w| w.iterable_plan_at(&iterable, source))
    }

    /// [`Walker::iterable_plan`] inside the iterated value's frame: the
    /// questions `S-For-Dyn` asks are about that value.
    fn iterable_plan_at(
        &mut self,
        iterable: &Desc,
        source: Option<&ValueKey>,
    ) -> Result<ForPlan, Reject> {
        match iterable {
            Desc::Composite(c) => {
                let count = c.elems.len();
                Ok(ForPlan {
                    elements: Elements::Ready(c.elems.clone()),
                    source: source.cloned(),
                    count,
                    extent: None,
                })
            }
            Desc::Array(a) => {
                let array = a.clone();
                let extent = *array.dims.first().ok_or_else(|| Reject::Unsupported {
                    construct: "iteration over a rank-0 array".to_string(),
                })?;
                let (count, guard) = match self.an.terms.constant_value(extent) {
                    Some(count) => (count.max(0) as usize, None),
                    None => {
                        // `S-For-Dyn` unrolls to the declared bound; phase 1
                        // has to answer whether the length is constant.
                        self.ctx()
                            .record_const(Rule::SForDyn.detail("the length"), extent);
                        (self.declared_bound(extent)?, Some(extent))
                    }
                };
                self.spend(count as u64)?;
                let keys = (0..count).map(|_| self.an.keys.next_expr()).collect();
                Ok(ForPlan {
                    elements: Elements::Slots { array, keys },
                    source: source.cloned(),
                    count,
                    extent: guard,
                })
            }
            other => Err(Reject::Unsupported {
                construct: format!("iteration over a {}", other.kind_str()),
            }),
        }
    }

    /// The elements a descriptor yields when it is iterated or unpacked: a
    /// composite's elements, or an array's rank−1 sub-descriptors along axis
    /// 0 (§5.11's `S-Unpack`, §6.2's `S-For-Elems`).
    ///
    /// Unlike a `for`, an unpacking binds every element at once and under one
    /// path condition, so the descriptors are built here rather than per
    /// iteration.
    pub fn iterable_elements(
        &mut self,
        d: &Desc,
        source: Option<&ValueKey>,
        context: &str,
    ) -> Result<Vec<Element>, Reject> {
        let plan = self.iterable_plan(d, source).map_err(|reject| match reject {
            Reject::Unsupported { construct } => Reject::Unsupported {
                construct: format!("{} ({})", construct, context),
            },
            other => other,
        })?;
        (0..plan.count).map(|j| plan.element(self, j)).collect()
    }

    /// One sub-descriptor of `a` along axis 0, under the key §5a rule 4
    /// allocated for it, with its entry in $\Delta$. A rank-1 array yields a
    /// data scalar; a higher-rank one yields a view of rank `r − 1`.
    fn array_element(
        &mut self,
        a: &ArrayDesc,
        key: &ValueKey,
        source: Option<&ValueKey>,
    ) -> Result<Desc, Reject> {
        let a = a.clone();
        let (desc, frame) = self.in_value(key, |w| {
            if let Some(source) = source {
                w.an.operand(source);
            }
            let tail: Vec<TermId> = a.dims[1..].to_vec();
            let tt = w.ctx().tt();
            if tail.is_empty() {
                let range = VarRange::of_dtype(&mut w.an.terms, a.dtype);
                return Ok(w
                    .ctx()
                    .data_scalar(Rule::EIndex, a.dtype, "elem", key, range, tt));
            }
            Ok(w.ctx().array(Rule::EIndex, a.dtype, tail, tt))
        })?;
        let producer = frame.into_producer(&desc);
        self.record(key.clone(), None, &desc, producer);
        Ok(desc)
    }

    /// The declared bound `B` of `S-For-Dyn`: the upper bound `Φ` already
    /// carries for the extent's variable, which comes from the input
    /// annotation (`N<=K`). Without one the trip count is unbounded and the
    /// loop cannot be unrolled.
    fn declared_bound(&mut self, extent: TermId) -> Result<usize, Reject> {
        let bound = self
            .upper_bound_of(extent)
            .ok_or(Reject::NonConstantRangeBound {
                detail: "iteration over an array whose length has no declared maximum".to_string(),
            })?;
        usize::try_from(bound).map_err(|_| Reject::NonConstantRangeBound {
            detail: format!("declared maximum {} is not a usable trip count", bound),
        })
    }

    /// The constant upper bound `Φ` records for a term that is one variable.
    pub fn upper_bound_of(&self, term: TermId) -> Option<i128> {
        let var = self.an.terms.var_of(term)?;
        let hi = self.an.phi.var_info(var).hi?;
        self.an.terms.constant_value(hi)
    }

    /// The compile-time integer value of a descriptor, when it has one.
    pub fn constant_of(&self, d: &Desc) -> Option<i128> {
        let scalar = d.as_scalar()?;
        match scalar.term {
            ScalarTerm::Int(t) => self.an.terms.constant_value(t),
            // A Boolean literal is an integer flag as far as `keepdims=` and
            // `return_inverse=` are concerned.
            ScalarTerm::Bool(b) if self.an.terms.is_true(b) => Some(1),
            ScalarTerm::Bool(b) if self.an.terms.is_false(b) => Some(0),
            _ => None,
        }
    }
}
