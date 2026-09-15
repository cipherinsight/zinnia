//! Statements: `S-Assign` and `S-Call` (§5.10), `S-Unpack` and `S-SetItem`
//! (§5.11), `S-If` (§6.1), `S-Return` and `S-Assert` (`coverage.md` §3).

use crate::ast::{
    ASTAssertStatement, ASTAssignStatement, ASTCondStatement, ASTNode, ASTReturnStatement,
};
use crate::phase0::desc::{
    Arm, CompositeKind, Desc, ValueKey,
};
use crate::phase0::rules::{e_composite, merge_desc};
use crate::phase0::source::Producer;
use crate::phase0::Reject;

use super::env::{merge_scopes, Returned, Scope};
use super::loops::Element;
use super::{created_at, node_name, Flow, Walker};
use crate::phase0::rules::Rule;

/// `S-Assign` (§5.10): `x = e`.
///
/// The value is evaluated once and bound to every target, left to right, as
/// Python evaluates a chained assignment.
pub(super) fn assign(w: &mut Walker, n: &ASTAssignStatement) -> Result<Flow, Reject> {
    let (value, key) = w.expr_keyed(&n.value)?;
    for target in &n.targets {
        w.while_cap.note_alias(target, &n.value);
        w.assign_target(target, value.clone(), Some(&key))?;
    }
    Ok(Flow::Normal)
}

/// `S-If` (§6.1).
///
/// A condition that folds to a constant takes one arm and its bindings
/// propagate unconditionally, which is what `visit_cond` does for a static
/// condition (`coverage.md` §3) and what keeps a dead arm's unsupported
/// constructs from rejecting the program.
///
/// Each arm is walked inside its own [`Arm`] component of the statement path,
/// so that the two arms' values get distinct keys even when a constant
/// condition means only one arm is walked (`implementation.md` §5a).
pub(super) fn cond(w: &mut Walker, n: &ASTCondStatement) -> Result<Flow, Reject> {
    let key = w.an.keys.next_expr();
    let guard = w.expr_at(&n.cond, &key)?;
    let b = w.condition(&guard, &key)?;
    if w.an.terms.is_true(b) {
        return arm(w, Arm::Then, |w| w.block(&n.t_block));
    }
    if w.an.terms.is_false(b) {
        return arm(w, Arm::Else, |w| w.block(&n.f_block));
    }
    // A dynamic `if` copies the scope into each arm and merges it after, so
    // it costs three passes over the bindings; charge them before the copies
    // so that a loop nest of merges cannot run away (`Walker::spend`).
    let bindings: u64 = w
        .frame()
        .names
        .values()
        .map(|binding| super::exprs::leaves(&binding.desc))
        .sum();
    w.spend(3 * bindings)?;
    let before = w.frame().names.clone();
    let then_flow = arm(w, Arm::Then, |w| w.under_branch(b, |w| w.block(&n.t_block)))?;
    let then_scope = std::mem::replace(&mut w.frame_mut().names, before);
    let not_b = w.an.terms.not(b);
    let else_flow = arm(w, Arm::Else, |w| {
        w.under_branch(not_b, |w| w.block(&n.f_block))
    })?;
    let else_scope = std::mem::take(&mut w.frame_mut().names);
    merge_branches(w, b, then_scope, else_scope)?;
    Ok(
        if then_flow == Flow::Terminated && else_flow == Flow::Terminated {
            Flow::Terminated
        } else {
            Flow::Normal
        },
    )
}

/// Run `body` inside the given arm of the key cursor's statement path.
fn arm<T>(
    w: &mut Walker,
    which: Arm,
    body: impl FnOnce(&mut Walker) -> Result<T, Reject>,
) -> Result<T, Reject> {
    let mark = w.an.keys.enter_arm(which);
    let out = body(w);
    w.an.keys.exit_arm(mark);
    out
}

/// `Env' = Env1 ⊔_b Env2` on the current frame (§6.1).
fn merge_branches(
    w: &mut Walker,
    b: crate::phase0::term::BoolId,
    then_scope: Scope,
    else_scope: Scope,
) -> Result<(), Reject> {

    merge_scopes(w, b, then_scope, else_scope)
}

/// `S-Return` (`coverage.md` §3): the returned value joins the frame's
/// delivery list under the path condition it was taken at; §6.1's `ite` merge
/// combines them when the frame ends.
pub(super) fn return_stmt(w: &mut Walker, n: &ASTReturnStatement) -> Result<Flow, Reject> {
    let returned_value = match &n.expr {
        Some(expr) => {
            let (desc, key) = w.expr_keyed(expr)?;
            Element::new(desc, Some(key))
        }
        None => Element::new(Desc::None, None),
    };
    let path = w.path();
    let returned = match w.frame().returned {
        None => path,
        Some(previous) => w.an.terms.or(previous, path),
    };
    let frame = w.frame_mut();
    frame.returns.push(Returned {
        value: returned_value,
        path,
    });
    frame.returned = Some(returned);
    Ok(Flow::Terminated)
}

/// `S-Assert` (`coverage.md` §3): `Φ += (φ ⇒ e)`, plus the CANFAIL obligation
/// that [`crate::phase0::rules::RuleCtx::check`] records with it.
pub(super) fn assert_stmt(w: &mut Walker, n: &ASTAssertStatement) -> Result<Flow, Reject> {
    let key = w.an.keys.next_expr();
    let desc = w.expr_at(&n.expr, &key)?;
    let claim = w.condition(&desc, &key)?;
    // The condition is the value the check is about (`implementation.md` §5a).
    w.with_subject(&key, |w| {
        w.ctx().check(Rule::SAssert, claim);
        Ok(())
    })?;
    Ok(Flow::Normal)
}

/// Merge the returns of a frame into one descriptor (§5.10: "several returns
/// merge by the rule `S-If`").
///
/// The fold is the baseline's: start from the first return and select each
/// later one under its own condition. The conditions are mutually exclusive,
/// because each excludes the ones before it, so the order does not matter for
/// the meaning — only for the shape of the term.
pub(crate) fn merge_returns(
    w: &mut Walker,
    returns: Vec<Returned>,
    key: &ValueKey,
    base: &str,
) -> Result<MergedReturns, Reject> {
    let operands: Vec<ValueKey> = returns
        .iter()
        .filter_map(|returned| returned.value.key.clone())
        .collect();
    // The merge is a fold, so the condition of the outermost `ite` — the one
    // the producer records — is the path condition of the last return. Each
    // earlier return contributes its own, which Δ does not repeat
    // (`implementation.md` §5a).
    let condition = returns.last().filter(|_| returns.len() > 1).map(|r| r.path);
    let mut iter = returns.into_iter();
    let Some(first) = iter.next() else {
        return Ok(MergedReturns {
            desc: Desc::None,
            operands,
            condition,
        });
    };
    let mut desc = first.value.desc;
    for (index, returned) in iter.enumerate() {
        let name = format!("{}_{}", base, index);
        desc = merge_desc(
            &mut w.ctx(),
            returned.path,
            &returned.value.desc,
            &desc,
            key,
            &name,
        )?;
    }
    Ok(MergedReturns {
        desc,
        operands,
        condition,
    })
}

/// What [`merge_returns`] produced: the merged value, the keys of the values
/// that were returned, and the condition of the outermost merge. The rules it
/// applied land in the value frame the caller has open.
pub(crate) struct MergedReturns {
    pub desc: Desc,
    pub operands: Vec<ValueKey>,
    pub condition: Option<crate::phase0::term::BoolId>,
}

impl MergedReturns {
    /// The producer of the merged value: one return is that value under a new
    /// key, several are a merge (`implementation.md` §5a).
    pub fn producer(&self, rules: Vec<Rule>) -> Producer {
        match (rules.is_empty(), self.operands.as_slice()) {
            (true, [only]) => Producer::Alias(only.clone()),
            _ => Producer::Applied {
                rules,
                operands: self.operands.clone(),
                condition: self.condition,
            },
        }
    }
}

impl Walker<'_> {
    /// Bind one assignment target (`do_recursive_assign`).
    ///
    /// Every binding merges against the previous one under the *liveness*
    /// condition, which is the baseline's `conditional_select`. Where the
    /// statement certainly runs the liveness folds to `true` and
    /// [`Walker::bind_name`] skips the merge, so an ordinary assignment and a
    /// loop target behave alike; §6.2's guard is what makes them differ,
    /// because an iteration the guard may skip has to leave the name at its
    /// previous value — the loop variable of a bounded `for` included.
    pub fn assign_target(
        &mut self,
        target: &ASTNode,
        value: Desc,
        source: Option<&ValueKey>,
    ) -> Result<(), Reject> {
        match target {
            ASTNode::ASTNameAssignTarget(t) => self.bind_name(&t.name, value, source),
            ASTNode::ASTTupleAssignTarget(t) => self.unpack(&t.targets, value, source),
            ASTNode::ASTListAssignTarget(t) => self.unpack(&t.targets, value, source),
            ASTNode::ASTSubscriptAssignTarget(t) => self.setitem(t, value, source),
            other => Err(Reject::Unsupported {
                construct: format!("{} as an assignment target", node_name(other)),
            }),
        }
    }

    /// `Env[x] ← D` (§5.10).
    ///
    /// When the statement runs only on some paths — after a conditional
    /// `break`, `continue` or `return` — the new binding is merged against
    /// the old one under the liveness condition, which is what the baseline's
    /// `select_value` does at `do_recursive_assign`. Inside an `if` the merge
    /// at the end of the branch (§6.1) already does this, so only the
    /// liveness part is applied here.
    ///
    /// The binding is a new SSA value, so it takes the $\psi$ of the state it
    /// is made in (§2), whatever the right-hand side carried. Most of the
    /// time the two agree, because the right-hand side was evaluated at the
    /// same path condition; they differ when the statement is a plain alias —
    /// `CA = CB` inside a branch hands on an input's descriptor, whose
    /// $\psi$ is `true` — and then $\Delta$ would claim the binding exists on
    /// every path.
    fn bind_name(
        &mut self,
        name: &str,
        value: Desc,
        source: Option<&ValueKey>,
    ) -> Result<(), Reject> {
        // One key per bound name, used both for the merge's fresh variable
        // and for the name's entry in Δ ("Value keys, as implemented").
        let key = self.an.keys.next_expr();
        let live = self.live();
        let unconditional = self.an.terms.is_true(live);
        let previous = self.frame().binding(name).cloned();
        let merged = !unconditional && previous.is_some();
        let (bound, frame) = self.in_value(&key, |w| match (merged, previous.as_ref()) {
            (true, Some(old)) => merge_desc(&mut w.ctx(), live, &value, &old.desc, &key, name),
            _ => Ok(value),
        })?;
        let path = self.path();
        let bound = created_at(bound, path);
        // `S-Assign` binds the name to the right-hand side; a binding the
        // liveness condition guards also merges against the previous version
        // (`implementation.md` §5a).
        let mut rules = vec![Rule::SAssign];
        rules.extend(frame.into_rules());
        let mut operands: Vec<ValueKey> = source.into_iter().cloned().collect();
        if merged {
            operands.extend(previous.as_ref().and_then(|old| old.key.clone()));
        }
        let producer = Producer::Applied {
            rules,
            operands,
            condition: merged.then_some(live),
        };
        self.record(key.clone(), Some(name), &bound, producer);
        self.frame_mut().set(name, bound, Some(key));
        Ok(())
    }

    /// `S-Unpack` (§5.11): `t_1, …, t_k = v`.
    ///
    /// The source is a composite, or an array unpacked along axis 0 into its
    /// rank−1 sub-descriptors. A starred target collects the middle; any
    /// other arity mismatch is a static error.
    fn unpack(
        &mut self,
        targets: &[ASTNode],
        value: Desc,
        source: Option<&ValueKey>,
    ) -> Result<(), Reject> {
        let elements = self.iterable_elements(&value, source, "unpacking")?;
        let star = targets.iter().position(is_starred);
        match star {
            None => {
                if targets.len() != elements.len() {
                    return Err(Reject::UnpackArity {
                        expected: targets.len(),
                        found: elements.len(),
                    });
                }
                for (target, element) in targets.iter().zip(elements) {
                    self.assign_target(target, element.desc, element.key.as_ref())?;
                }
                Ok(())
            }
            Some(at) => self.unpack_starred(targets, at, elements, source),
        }
    }

    /// The starred form of `S-Unpack`: `a, *b, c = v`.
    fn unpack_starred(
        &mut self,
        targets: &[ASTNode],
        star: usize,
        elements: Vec<Element>,
        source: Option<&ValueKey>,
    ) -> Result<(), Reject> {
        let before = star;
        let after = targets.len() - star - 1;
        if elements.len() < before + after {
            return Err(Reject::UnpackArity {
                expected: before + after,
                found: elements.len(),
            });
        }
        let middle = elements.len() - before - after;
        for (index, target) in targets.iter().enumerate().take(before) {
            let element = elements[index].clone();
            self.assign_target(target, element.desc, element.key.as_ref())?;
        }
        // The collected middle is a new composite; its elements are named by
        // the unpacked value, so that is the source it points at.
        let collected = elements[before..before + middle]
            .iter()
            .map(|element| element.desc.clone())
            .collect();
        self.ctx().applied(Rule::SUnpack);
        self.assign_target(
            &targets[star],
            e_composite(CompositeKind::List, collected),
            source,
        )?;
        for index in 0..after {
            let element = elements[before + middle + index].clone();
            self.assign_target(
                &targets[star + 1 + index],
                element.desc,
                element.key.as_ref(),
            )?;
        }
        Ok(())
    }
}

/// A starred target carries the flag on the target node itself
/// (`zinnia/compile/transformer/base.py`'s `visit_assign_target`).
fn is_starred(target: &ASTNode) -> bool {
    match target {
        ASTNode::ASTNameAssignTarget(t) => t.star,
        ASTNode::ASTSubscriptAssignTarget(t) => t.star,
        ASTNode::ASTTupleAssignTarget(t) => t.star,
        ASTNode::ASTListAssignTarget(t) => t.star,
        _ => false,
    }
}
