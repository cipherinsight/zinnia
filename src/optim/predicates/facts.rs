//! Fact-propagation framework: per-SSA-value `Bool` fact storage with
//! control-flow-aware scoping.
//!
//! ## Overview
//!
//! A **fact** is a Bool-typed [`ContractTerm`] whose leaves reference IR
//! values by [`StmtId`] (or by input name, for `@requires`-seeded facts).
//! Once a fact lands in a [`FactSet`], its meaning is "this Bool is known
//! to hold in this scope, given the path conditions in effect."
//!
//! Facts are produced two ways:
//!
//! 1. **Seeding** — at function/chip entry, each `@requires` term becomes
//!    a free fact (no anchor; it references inputs by name).
//! 2. **Op contracts** — when an op fires at a specific IR statement, its
//!    `ensures` template gets instantiated (`Var(Output)` → `Var(SsaPtr(result))`,
//!    `Var(Input("argN"))` → `Var(SsaPtr(args[N]))`) and the resulting
//!    ptr-anchored Bool is inserted into the current scope's [`FactSet`].
//!
//! Facts are consumed by the resolver (see the `resolver-prove-api` card):
//! given a query Bool `q`, the resolver asserts the visible facts on its
//! solver and asks whether `q` follows.
//!
//! ## Control-flow rules
//!
//! - **Sequential**: insert into the top scope.
//! - **Conditional**: open a child scope with the branch predicate as its
//!   path condition. Facts produced inside are local. At the join, facts
//!   true on **both** arms survive in the parent unconditionally; arm-only
//!   facts survive as guarded implications via [`merge_branches`].
//! - **Loop**: opening a loop scope captures the pre-loop fact snapshot;
//!   on exit, facts about per-iteration values (those produced *inside* the
//!   loop) are dropped. Facts inherited from parent scopes survive.
//! - **Chip/function boundary**: entry seeds from `@requires`; exit exports
//!   the postcondition fact set for the parent chip's consumption.

use std::collections::{HashMap, HashSet};

use crate::optim::predicates::formula::{BoolOp, ContractTerm};
use crate::types::ValueId;

/// A single Bool fact. After instantiation its leaves are
/// `Var(Input(_))` or `Var(Value(_))` plus literals — never `Var(Output)`,
/// which only appears in untemplated contract `ensures` clauses.
pub type Fact = ContractTerm;

// ---------------------------------------------------------------------------
// FactSet — per-scope storage of facts
// ---------------------------------------------------------------------------

/// Facts visible in a single scope, before path-conditioning is applied at
/// scope exit.
///
/// All facts anchor on a [`ValueId`]
/// (compiler.value-id-and-fact-leaves): a fact about a Value lives under
/// that Value's `value_id` in `by_value`. Multi-value facts are indexed
/// at every value they reference. `@requires` seeds, which originally
/// arrive with `Var(Input(name))` leaves, are substituted to
/// `Var(Value(vid))` at intake against the parameter's `ReadInteger`
/// Value, then indexed normally.
#[derive(Debug, Default, Clone)]
pub struct FactSet {
    pub by_value: HashMap<ValueId, Vec<Fact>>,
}

impl FactSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a fact anchored to `vid` (typically the producing op's result).
    pub fn insert_for(&mut self, vid: ValueId, fact: Fact) {
        self.by_value.entry(vid).or_default().push(fact);
    }

    /// All facts visible in the set, flattened. May contain duplicates
    /// when the same fact body is indexed at multiple anchors. Consumers
    /// that need deduplication should hash on `ContractTerm` themselves.
    pub fn all(&self) -> Vec<&Fact> {
        self.by_value.values().flatten().collect()
    }

    /// Number of `(anchor, fact)` entries in this set. Duplicates of the
    /// same fact body indexed at distinct value_ids count separately.
    pub fn len(&self) -> usize {
        self.by_value.values().map(|v| v.len()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Additive merge: append everything from `other`. Anchors are preserved.
    pub fn extend(&mut self, other: FactSet) {
        for (vid, mut facts) in other.by_value {
            self.by_value.entry(vid).or_default().append(&mut facts);
        }
    }

    /// Intersection: keep only facts present in **both** sets (compared by
    /// structural [`ContractTerm`] equality at the same anchor). Anchors
    /// are preserved.
    pub fn intersect(&self, other: &FactSet) -> FactSet {
        let mut out = FactSet::new();
        for (vid, facts) in &self.by_value {
            if let Some(other_facts) = other.by_value.get(vid) {
                let other_set: HashSet<&Fact> = other_facts.iter().collect();
                for f in facts {
                    if other_set.contains(f) {
                        out.by_value.entry(*vid).or_default().push(f.clone());
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// FactScope — facts + path-condition + scope kind
// ---------------------------------------------------------------------------

/// Kinds of scope; semantics on exit differ per kind.
///
/// Mirrors the IR-gen scope stack ([`crate::scope::Scope`]) but only the
/// fields the fact framework cares about. The mapping is:
/// - `Master / Generator` → [`ScopeKind::Plain`] (sequential rule only).
/// - `Chip` → [`ScopeKind::Chip`] (function-boundary postcondition extraction).
/// - `Conditional` → [`ScopeKind::Conditional`] (branch rule).
/// - `Loop` → [`ScopeKind::Loop`] (loop rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Plain,
    Chip,
    Conditional,
    Loop,
}

/// A scope's facts plus its accumulated path condition.
///
/// The `path_conditions` field stores the **delta** added when this scope
/// was opened, not the cumulative path from the root — combining scopes is
/// the job of the [`FactStack`] consumers (typically the resolver).
#[derive(Debug, Clone)]
pub struct FactScope {
    pub kind: ScopeKind,
    /// Facts added in this scope.
    pub facts: FactSet,
    /// Path predicates added by entering this scope (e.g., the `if`
    /// condition or its negation). Empty for `Plain`/`Loop`/`Chip` scopes.
    pub path_conditions: Vec<Fact>,
}

impl FactScope {
    pub fn new(kind: ScopeKind) -> Self {
        Self {
            kind,
            facts: FactSet::new(),
            path_conditions: Vec::new(),
        }
    }

    pub fn with_path(mut self, predicate: Fact) -> Self {
        self.path_conditions.push(predicate);
        self
    }
}

// ---------------------------------------------------------------------------
// FactStack — the per-IR-gen propagation tracker
// ---------------------------------------------------------------------------

/// Stack of [`FactScope`]s mirroring the IR-gen scope nesting.
///
/// Lifecycle:
/// ```text
/// FactStack::new()  // bottom Plain scope auto-pushed
///   enter(...)      // push a child scope (Chip / Conditional / Loop / Plain)
///   insert(...)     // add facts to the top
///   leave()         // pop and return the scope's contents
///   merge_branches(then_scope, else_scope)  // fold two if-arms into parent
/// ```
///
/// The stack also keeps `per_value`, a flat per-`ValueId` archive of every
/// fact ever produced anywhere in the function. Downstream consumers
/// (resolver, prove API) read this when they need to look up "what did we
/// learn about Value v?" without worrying about which scope produced it.
#[derive(Debug)]
pub struct FactStack {
    scopes: Vec<FactScope>,
    /// Flat archive: value_id → all facts ever anchored to it. Survives
    /// scope exits so the resolver can query historical facts.
    pub per_value: HashMap<ValueId, Vec<Fact>>,
    /// Postcondition export from the most recently closed `Chip` scope.
    /// Read by the parent chip's IR-gen to inherit callee guarantees.
    pub last_chip_postcondition: Option<FactSet>,
}

impl Default for FactStack {
    fn default() -> Self {
        Self::new()
    }
}

impl FactStack {
    pub fn new() -> Self {
        Self {
            scopes: vec![FactScope::new(ScopeKind::Plain)],
            per_value: HashMap::new(),
            last_chip_postcondition: None,
        }
    }

    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    pub fn current(&self) -> &FactScope {
        self.scopes.last().expect("FactStack: empty stack")
    }

    pub fn current_mut(&mut self) -> &mut FactScope {
        self.scopes.last_mut().expect("FactStack: empty stack")
    }

    /// Open a new child scope of the given kind. For `Conditional`, pass
    /// the branch predicate via `path_predicate` (the caller is responsible
    /// for negating it for an `else`-arm); pass `None` otherwise.
    pub fn enter(&mut self, kind: ScopeKind, path_predicate: Option<Fact>) {
        let mut s = FactScope::new(kind);
        if let Some(p) = path_predicate {
            s.path_conditions.push(p);
        }
        self.scopes.push(s);
    }

    /// Close the top scope. For `Chip`, the closed scope's facts are also
    /// stashed in `last_chip_postcondition` for the parent to inherit.
    pub fn leave(&mut self) -> FactScope {
        assert!(self.scopes.len() > 1, "FactStack: cannot pop root scope");
        let popped = self.scopes.pop().unwrap();
        if popped.kind == ScopeKind::Chip {
            self.last_chip_postcondition = Some(popped.facts.clone());
        }
        popped
    }

    /// Insert a value_id-anchored fact at the current top. Also archives
    /// in `per_value` so historical lookups work after the scope is left.
    pub fn insert_for(&mut self, vid: ValueId, fact: Fact) {
        self.current_mut().facts.insert_for(vid, fact.clone());
        self.per_value.entry(vid).or_default().push(fact);
    }

    /// Merge two completed if-arms into the parent scope (conservative).
    ///
    /// Only facts present in **both** arms (compared by structural equality
    /// at the same anchor) survive. Arm-only facts are dropped.
    ///
    /// Use this when no usable path predicate is available — soundness is
    /// preserved by erring on the side of forgetting facts rather than
    /// claiming them unconditionally. Used by v1 IR-gen wiring;
    /// [`merge_branches`] is the richer version that handles guards.
    pub fn merge_branches_intersect_only(
        &mut self,
        then_scope: FactScope,
        else_scope: FactScope,
    ) {
        let common = then_scope.facts.intersect(&else_scope.facts);
        self.current_mut().facts.extend(common);
    }

    /// Merge two completed if-arms into the parent scope.
    ///
    /// `then_scope` and `else_scope` are the (already-popped) scopes for
    /// the `then` and `else` arms respectively. The merge:
    /// - facts in both arms → added to the parent's [`FactSet`] as-is,
    ///   preserving anchors,
    /// - then-only facts → wrapped as `p ⇒ fact`, **anchor preserved**,
    ///   indexed under every ptr the wrapped fact references,
    /// - else-only facts → wrapped as `¬p ⇒ fact`, anchor preserved.
    ///
    /// where `p` is the `then`-arm's path predicate (assumed to be a single
    /// Bool; multi-predicate scopes use the conjunction).
    ///
    /// (compiler.eliminate-free-fact-bucket) The arm-only facts used to
    /// drop into the parent's `free` bucket because the wrapping flattened
    /// the original anchor. They now stay in `by_ptr`: a wrapped fact
    /// about ptr `P` is still a claim about `P` — just one whose truth is
    /// conditional on `p`. Consumers querying `by_ptr[P]` find both
    /// unconditional and conditional claims; the SMT prover discharges
    /// `(p → f) ∧ p ⊢ f` automatically, and the scanner side
    /// ([`derive_bounds_from_facts`]) looks through `Implies` when the
    /// antecedent matches an in-scope path condition.
    pub fn merge_branches(&mut self, then_scope: FactScope, else_scope: FactScope) {
        let common = then_scope.facts.intersect(&else_scope.facts);
        self.current_mut().facts.extend(common);

        let p = conjunction(&then_scope.path_conditions);
        let not_p = ContractTerm::Not(Box::new(p.clone()));

        let then_only = subtract(&then_scope.facts, &else_scope.facts);
        for (anchors, f) in then_only {
            let wrapped = implies(p.clone(), f);
            // Anchor at every ValueId the original fact referenced. Falls
            // back to the wrapped term's own value_ids (typically a superset
            // of the original anchors) if the original was unanchored.
            let final_anchors: Vec<ValueId> = if anchors.is_empty() {
                collect_value_ids(&wrapped)
            } else {
                anchors
            };
            for anchor in &final_anchors {
                self.current_mut().facts.insert_for(*anchor, wrapped.clone());
            }
        }

        let else_only = subtract(&else_scope.facts, &then_scope.facts);
        for (anchors, f) in else_only {
            let wrapped = implies(not_p.clone(), f);
            let final_anchors: Vec<ValueId> = if anchors.is_empty() {
                collect_value_ids(&wrapped)
            } else {
                anchors
            };
            for anchor in &final_anchors {
                self.current_mut().facts.insert_for(*anchor, wrapped.clone());
            }
        }
    }

    /// Visit-all-facts: walk the live stack from root to top.
    pub fn visible_facts(&self) -> Vec<&Fact> {
        let mut out = Vec::new();
        for s in &self.scopes {
            out.extend(s.facts.all());
        }
        out
    }

    /// Visit-all-path-conditions on the live stack.
    pub fn visible_path_conditions(&self) -> Vec<&Fact> {
        let mut out = Vec::new();
        for s in &self.scopes {
            out.extend(s.path_conditions.iter());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Contract-instantiation helper
// ---------------------------------------------------------------------------

/// Walk `template` and substitute every `Var(Output)` with
/// `Var(SsaPtr(output))` and every `Var(Formal(name))` with
/// `Var(SsaPtr(formals_to_args[name]))`.
///
/// `Var(Input(name))` leaves are **not** substituted — they refer to user
/// chip-inputs by source-level name and are resolved at lowering time by
/// the resolver via [`Substitution::with_input`]. Mixing both in one
/// template is legal (e.g., a contract template that says
/// `Formal("len_out") == Input("k")` when the contract is wired into a
/// chip whose input is named `k`).
///
/// Unmapped `Formal(name)` leaves survive in the result and will surface
/// as [`LowerError::UnboundFormal`] when (if) the fact is lowered. Callers
/// should treat that as a malformed contract registration.
/// Walk `template` and substitute every `Var(Input(name))` whose name is
/// present in `inputs_to_values` with `Var(Value(vid))`. Used by the intake
/// `@requires` flow (compiler.value-id-and-fact-leaves) to bind
/// user-precondition leaves to the parameters' ValueIds before the fact
/// is anchored into `by_value`.
///
/// Names not present in the map are left as-is (the fact survives in
/// partially-bound form).
pub fn substitute_inputs(
    template: &ContractTerm,
    inputs_to_values: &HashMap<String, ValueId>,
) -> ContractTerm {
    use crate::optim::predicates::formula::ContractVar;
    match template {
        ContractTerm::Var(ContractVar::Input(name)) => match inputs_to_values.get(name) {
            Some(vid) => ContractTerm::Var(ContractVar::Value(*vid)),
            None => template.clone(),
        },
        ContractTerm::Var(_)
        | ContractTerm::LitInt(_)
        | ContractTerm::LitFloat(_)
        | ContractTerm::LitBool(_) => template.clone(),
        ContractTerm::Arith { op, lhs, rhs } => ContractTerm::Arith {
            op: *op,
            lhs: Box::new(substitute_inputs(lhs, inputs_to_values)),
            rhs: Box::new(substitute_inputs(rhs, inputs_to_values)),
        },
        ContractTerm::Cmp { op, lhs, rhs } => ContractTerm::Cmp {
            op: *op,
            lhs: Box::new(substitute_inputs(lhs, inputs_to_values)),
            rhs: Box::new(substitute_inputs(rhs, inputs_to_values)),
        },
        ContractTerm::BoolComb { op, operands } => ContractTerm::BoolComb {
            op: *op,
            operands: operands
                .iter()
                .map(|o| substitute_inputs(o, inputs_to_values))
                .collect(),
        },
        ContractTerm::Not(inner) => {
            ContractTerm::Not(Box::new(substitute_inputs(inner, inputs_to_values)))
        }
        ContractTerm::PredicateApp { kind, args } => ContractTerm::PredicateApp {
            kind: kind.clone(),
            args: args
                .iter()
                .map(|a| substitute_inputs(a, inputs_to_values))
                .collect(),
        },
    }
}

/// Collect every `Var(Value(vid))` leaf in `term`, deduplicated and order-
/// preserving (first-seen). Used to find every anchor a fact should be
/// indexed at (compiler.value-id-and-fact-leaves): facts about a Value
/// are stored under the Value's `value_id`, and a multi-value fact gets
/// indexed at every Value it mentions.
pub fn collect_value_ids(term: &ContractTerm) -> Vec<ValueId> {
    use crate::optim::predicates::formula::ContractVar;
    let mut out: Vec<ValueId> = Vec::new();
    fn walk(t: &ContractTerm, out: &mut Vec<ValueId>) {
        match t {
            ContractTerm::Var(ContractVar::Value(vid)) => {
                if !out.contains(vid) {
                    out.push(*vid);
                }
            }
            ContractTerm::Var(_)
            | ContractTerm::LitInt(_)
            | ContractTerm::LitFloat(_)
            | ContractTerm::LitBool(_) => {}
            ContractTerm::Arith { lhs, rhs, .. } | ContractTerm::Cmp { lhs, rhs, .. } => {
                walk(lhs, out);
                walk(rhs, out);
            }
            ContractTerm::BoolComb { operands, .. } => {
                for o in operands {
                    walk(o, out);
                }
            }
            ContractTerm::Not(inner) => walk(inner, out),
            ContractTerm::PredicateApp { args, .. } => {
                for a in args {
                    walk(a, out);
                }
            }
        }
    }
    walk(term, &mut out);
    out
}

pub fn instantiate_contract(
    template: &ContractTerm,
    output: Option<ValueId>,
    formals_to_args: &HashMap<String, ValueId>,
) -> ContractTerm {
    use crate::optim::predicates::formula::ContractVar;

    match template {
        ContractTerm::Var(ContractVar::Output) => match output {
            Some(vid) => ContractTerm::Var(ContractVar::Value(vid)),
            None => template.clone(),
        },
        ContractTerm::Var(ContractVar::Formal(name)) => match formals_to_args.get(name) {
            Some(vid) => ContractTerm::Var(ContractVar::Value(*vid)),
            None => template.clone(),
        },
        ContractTerm::Var(ContractVar::Input(_))
        | ContractTerm::Var(ContractVar::Value(_))
        | ContractTerm::LitInt(_)
        | ContractTerm::LitFloat(_)
        | ContractTerm::LitBool(_) => template.clone(),
        ContractTerm::Arith { op, lhs, rhs } => ContractTerm::Arith {
            op: *op,
            lhs: Box::new(instantiate_contract(lhs, output, formals_to_args)),
            rhs: Box::new(instantiate_contract(rhs, output, formals_to_args)),
        },
        ContractTerm::Cmp { op, lhs, rhs } => ContractTerm::Cmp {
            op: *op,
            lhs: Box::new(instantiate_contract(lhs, output, formals_to_args)),
            rhs: Box::new(instantiate_contract(rhs, output, formals_to_args)),
        },
        ContractTerm::BoolComb { op, operands } => ContractTerm::BoolComb {
            op: *op,
            operands: operands
                .iter()
                .map(|o| instantiate_contract(o, output, formals_to_args))
                .collect(),
        },
        ContractTerm::Not(inner) => {
            ContractTerm::Not(Box::new(instantiate_contract(inner, output, formals_to_args)))
        }
        ContractTerm::PredicateApp { kind, args } => ContractTerm::PredicateApp {
            kind: kind.clone(),
            args: args
                .iter()
                .map(|a| instantiate_contract(a, output, formals_to_args))
                .collect(),
        },
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn conjunction(preds: &[Fact]) -> Fact {
    match preds.len() {
        0 => ContractTerm::LitBool(true),
        1 => preds[0].clone(),
        _ => ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: preds.to_vec(),
        },
    }
}

/// `lhs ⇒ rhs`, encoded as `(¬lhs) ∨ rhs`.
fn implies(lhs: Fact, rhs: Fact) -> Fact {
    ContractTerm::BoolComb {
        op: BoolOp::Or,
        operands: vec![ContractTerm::Not(Box::new(lhs)), rhs],
    }
}

/// Facts in `a` not present in `b` (by structural equality), each paired
/// with the anchor it was indexed under in `a`. Multi-anchor facts (the
/// same body indexed at several ptrs) appear once per anchor; the merge
/// step re-indexes them at the same anchors after wrapping.
fn subtract(a: &FactSet, b: &FactSet) -> Vec<(Vec<ValueId>, Fact)> {
    let mut out: Vec<(Vec<ValueId>, Fact)> = Vec::new();
    for (vid, facts) in &a.by_value {
        let b_set: HashSet<&Fact> = b
            .by_value
            .get(vid)
            .map(|v| v.iter().collect())
            .unwrap_or_default();
        for f in facts.iter().filter(|f| !b_set.contains(f)) {
            out.push((vec![*vid], f.clone()));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::predicates::formula::{CmpOp, ContractVar};

    fn lit_int(n: i64) -> ContractTerm {
        ContractTerm::LitInt(n)
    }
    fn vid(n: u64) -> ValueId {
        ValueId(n)
    }
    fn vref(n: u64) -> ContractTerm {
        ContractTerm::Var(ContractVar::Value(vid(n)))
    }
    fn input(n: &str) -> ContractTerm {
        ContractTerm::Var(ContractVar::Input(n.to_string()))
    }
    fn cmp(op: CmpOp, lhs: ContractTerm, rhs: ContractTerm) -> ContractTerm {
        ContractTerm::Cmp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    // ── FactSet basics ────────────────────────────────────────────────

    #[test]
    fn factset_insert_and_query() {
        let mut s = FactSet::new();
        let f1 = cmp(CmpOp::Ge, vref(7), lit_int(0));
        let f2 = cmp(CmpOp::Eq, vref(3), lit_int(3));
        s.insert_for(vid(7), f1.clone());
        s.insert_for(vid(3), f2.clone());
        assert_eq!(s.len(), 2);
        let all = s.all();
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|x| **x == f1));
    }

    #[test]
    fn factset_intersect_keeps_only_common() {
        let mut a = FactSet::new();
        let mut b = FactSet::new();
        let f1 = cmp(CmpOp::Ge, vref(1), lit_int(0));
        let f2 = cmp(CmpOp::Le, vref(1), lit_int(10));
        let f3 = cmp(CmpOp::Eq, vref(2), lit_int(5));
        a.insert_for(vid(1), f1.clone());
        a.insert_for(vid(1), f2.clone());
        b.insert_for(vid(1), f1.clone());
        b.insert_for(vid(2), f3.clone());
        let common = a.intersect(&b);
        // Only f1 is in both arms at the same anchor.
        assert_eq!(common.len(), 1);
        assert_eq!(common.by_value.get(&vid(1)).unwrap()[0], f1);
    }

    #[test]
    fn factset_extend_is_additive() {
        let mut a = FactSet::new();
        let mut b = FactSet::new();
        a.insert_for(vid(1), cmp(CmpOp::Ge, vref(1), lit_int(0)));
        b.insert_for(vid(1), cmp(CmpOp::Le, vref(1), lit_int(10)));
        b.insert_for(vid(2), cmp(CmpOp::Ge, vref(2), lit_int(0)));
        a.extend(b);
        assert_eq!(a.len(), 3);
    }

    // ── FactStack: sequential rule ────────────────────────────────────

    #[test]
    fn stack_sequential_appends_to_top() {
        let mut stk = FactStack::new();
        stk.insert_for(vid(1), cmp(CmpOp::Eq, vref(1), lit_int(0)));
        stk.insert_for(vid(2), cmp(CmpOp::Eq, vref(2), lit_int(1)));
        assert_eq!(stk.current().facts.len(), 2);
        assert_eq!(stk.per_value.get(&vid(1)).unwrap().len(), 1);
        assert_eq!(stk.per_value.get(&vid(2)).unwrap().len(), 1);
    }

    // ── FactStack: branch join ────────────────────────────────────────

    #[test]
    fn stack_branch_join_intersects_common_facts() {
        let mut stk = FactStack::new();
        let pred = cmp(CmpOp::Gt, input("k"), lit_int(0));

        // then-arm
        stk.enter(ScopeKind::Conditional, Some(pred.clone()));
        stk.insert_for(vid(10), cmp(CmpOp::Ge, vref(10), lit_int(0)));
        stk.insert_for(vid(10), cmp(CmpOp::Le, vref(10), lit_int(100)));
        let then_scope = stk.leave();

        // else-arm
        stk.enter(ScopeKind::Conditional, None);
        stk.insert_for(vid(10), cmp(CmpOp::Ge, vref(10), lit_int(0)));
        stk.insert_for(vid(10), cmp(CmpOp::Eq, vref(10), lit_int(-1)));
        let else_scope = stk.leave();

        stk.merge_branches(then_scope, else_scope);

        // The `Ge 0` fact appears in both arms → survives at anchor 10
        // unwrapped. The `Le 100` (then-only) and `Eq -1` (else-only)
        // appear at the same anchor as **guarded implications**.
        let by_value = stk.current().facts.by_value.get(&vid(10)).unwrap();
        assert!(by_value
            .iter()
            .any(|f| *f == cmp(CmpOp::Ge, vref(10), lit_int(0))));
        let guarded_count = by_value
            .iter()
            .filter(|f| matches!(f, ContractTerm::BoolComb { op: BoolOp::Or, .. }))
            .count();
        assert_eq!(
            guarded_count, 2,
            "expected two guarded implications at value_id 10; got {:?}",
            by_value
        );
    }

    // ── FactStack: loop drop ─────────────────────────────────────────

    #[test]
    fn stack_loop_scope_drops_local_facts_on_exit() {
        let mut stk = FactStack::new();
        stk.insert_for(vid(1), cmp(CmpOp::Ge, vref(1), lit_int(0)));
        stk.enter(ScopeKind::Loop, None);
        stk.insert_for(vid(2), cmp(CmpOp::Ge, vref(2), lit_int(0)));
        assert_eq!(stk.visible_facts().len(), 2);
        let _ = stk.leave();
        assert_eq!(stk.visible_facts().len(), 1);
        assert!(stk
            .visible_facts()
            .iter()
            .any(|f| **f == cmp(CmpOp::Ge, vref(1), lit_int(0))));
    }

    // ── FactStack: chip boundary ─────────────────────────────────────

    #[test]
    fn stack_chip_exit_exports_postcondition() {
        let mut stk = FactStack::new();
        stk.enter(ScopeKind::Chip, None);
        stk.insert_for(vid(42), cmp(CmpOp::Eq, vref(42), lit_int(7)));
        let _ = stk.leave();
        let post = stk.last_chip_postcondition.as_ref().unwrap();
        assert_eq!(post.len(), 1);
        assert_eq!(
            post.by_value.get(&vid(42)).unwrap()[0],
            cmp(CmpOp::Eq, vref(42), lit_int(7))
        );
    }

    // ── instantiate_contract ─────────────────────────────────────────

    #[test]
    fn instantiate_substitutes_output_and_known_formals() {
        let template = cmp(
            CmpOp::Eq,
            ContractTerm::Var(ContractVar::Output),
            ContractTerm::Var(ContractVar::Formal("arg0".to_string())),
        );
        let mut formals: HashMap<String, ValueId> = HashMap::new();
        formals.insert("arg0".to_string(), vid(5));

        let inst = instantiate_contract(&template, Some(vid(99)), &formals);
        let expected = cmp(CmpOp::Eq, vref(99), vref(5));
        assert_eq!(inst, expected);
    }

    #[test]
    fn instantiate_leaves_inputs_by_name() {
        let template = cmp(
            CmpOp::Eq,
            ContractTerm::Var(ContractVar::Output),
            input("k"),
        );
        let formals: HashMap<String, ValueId> = HashMap::new();
        let inst = instantiate_contract(&template, Some(vid(11)), &formals);
        let expected = cmp(CmpOp::Eq, vref(11), input("k"));
        assert_eq!(inst, expected);
    }

    #[test]
    fn instantiate_leaves_unmapped_formals_in_place() {
        let template = cmp(
            CmpOp::Eq,
            ContractTerm::Var(ContractVar::Output),
            ContractTerm::Var(ContractVar::Formal("missing".to_string())),
        );
        let formals: HashMap<String, ValueId> = HashMap::new();
        let inst = instantiate_contract(&template, Some(vid(11)), &formals);
        let expected = cmp(
            CmpOp::Eq,
            vref(11),
            ContractTerm::Var(ContractVar::Formal("missing".to_string())),
        );
        assert_eq!(inst, expected);
    }
}
