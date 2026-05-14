//! `prove` — ad-hoc Bool query API for op-contract consumers.
//!
//! Given the live [`FactStack`] (held by [`IRBuilder`]), [`prove`] answers
//! "does the supplied `ContractTerm` follow from the visible facts and
//! path conditions?" with one of three outcomes:
//!
//! - [`ProveOutcome::Proved`] — the SMT layer showed `(facts ∧ paths ∧ ¬term)`
//!   is unsatisfiable. The term is entailed.
//! - [`ProveOutcome::Disproved`] — `(facts ∧ paths ∧ term)` is unsatisfiable.
//!   The term contradicts the facts.
//! - [`ProveOutcome::Unknown`] — neither check succeeded, or the SMT layer
//!   timed out. Callers MUST treat this as "no information" — never as Proved.
//!
//! ## Soundness rule
//!
//! Treating `Unknown` as `Proved` is a circuit-correctness bug. Ops that
//! call `prove` to pick a fast path must fall back to the general path on
//! anything other than `Proved`.
//!
//! ## What `prove` does NOT do
//!
//! - It does not consult the IR's computational structure (def-use chains
//!   of ops in the body). Facts produced by op contracts encode the
//!   semantics; the resolver-side IR walking is reserved for the existing
//!   `resolve_*` methods. This keeps `prove` cheap and local: one fresh
//!   solver per query, only the FactStack as input.
//! - It does not cache across queries (v1). A per-query cache fits the
//!   call site's needs since ops typically call `prove` once at codegen
//!   time and re-asking is rare. Adding a `(term, fact-set, paths)`-keyed
//!   cache later is straightforward; see the card README for the design.

use std::collections::HashMap;

use z3::ast::{Ast, Bool, Int};

use crate::builder::IRBuilder;
use crate::optim::predicates::facts::{Fact, FactScope};
use crate::optim::predicates::formula::{self, ContractTerm, ContractVar, Substitution};
use crate::types::ValueId;

// ---------------------------------------------------------------------------
// ProveOutcome
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProveOutcome {
    /// `(facts ∧ paths) ⇒ term` was discharged by SMT.
    Proved,
    /// `(facts ∧ paths) ⇒ ¬term` was discharged by SMT.
    Disproved,
    /// No discharge succeeded — could mean timeout, no entailment, or no
    /// contradiction. Treat as "no information."
    Unknown,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Query whether `term` follows from the visible facts + path conditions
/// currently held by `b.facts`. See module docs for outcome semantics.
///
/// Reads `ZINNIA_SMT_PROVE_TIMEOUT_MS` for the per-check budget; defaults
/// to 1000 ms. The same budget applies to both the entailment check and
/// (if needed) the contradiction check.
pub fn prove(b: &IRBuilder, term: &ContractTerm) -> ProveOutcome {
    // A/B-harness kill switch: under `ZINNIA_REQ_DISABLE=1` the prove
    // layer returns Unknown unconditionally. The discharge_requires
    // lenient branch then emits the witness check (`IR::Assert`), so
    // preconditions are still enforced at proof time — the soundness
    // floor is intact. See `compiler.verification-ab-disable-harness`.
    if crate::optim::resolver::req_disabled() {
        return ProveOutcome::Unknown;
    }
    let facts = b.facts.visible_facts().into_iter().cloned().collect::<Vec<Fact>>();
    let paths = b
        .facts
        .visible_path_conditions()
        .into_iter()
        .cloned()
        .collect::<Vec<Fact>>();
    prove_with_facts(term, &facts, &paths)
}

/// Same as [`prove`] but takes an explicit fact list and path-condition
/// list. Useful for tests and for deferred queries against a stashed
/// [`FactScope`].
pub fn prove_with_facts(term: &ContractTerm, facts: &[Fact], paths: &[Fact]) -> ProveOutcome {
    let timeout_ms = std::env::var("ZINNIA_SMT_PROVE_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1000);

    let entails = check_unsat(facts, paths, term, /* negate */ true, timeout_ms);
    match entails {
        Some(true) => return ProveOutcome::Proved,
        Some(false) | None => {}
    }
    let contradicts = check_unsat(facts, paths, term, /* negate */ false, timeout_ms);
    match contradicts {
        Some(true) => ProveOutcome::Disproved,
        _ => ProveOutcome::Unknown,
    }
}

/// Convenience for [`prove`] specifically against a stashed scope —
/// equivalent to running prove with that scope's facts and path conditions
/// only, ignoring whatever is currently live on the stack. Useful when
/// the consumer wants to query a closed scope (e.g., a chip's exported
/// postcondition).
pub fn prove_in_scope(scope: &FactScope, term: &ContractTerm) -> ProveOutcome {
    let facts: Vec<Fact> = scope.facts.all().into_iter().cloned().collect();
    prove_with_facts(term, &facts, &scope.path_conditions)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Assert `facts ∧ paths ∧ (¬term | term)` on a fresh solver and check sat.
///
/// `negate = true`  → assert `¬term`; unsat means term is *entailed*.
/// `negate = false` → assert `term`;  unsat means term is *contradicted*.
///
/// Returns `Some(true)` if unsat (the chosen relation held), `Some(false)`
/// if sat, `None` if Z3 returned Unknown (typically a timeout).
fn check_unsat(
    facts: &[Fact],
    paths: &[Fact],
    term: &ContractTerm,
    negate: bool,
    timeout_ms: u64,
) -> Option<bool> {
    // Build a single Substitution over every Input / Value(vid) the
    // assembled formula references. Fresh symbolic Ints per name/value_id;
    // the SMT layer then sees the names by string identity (same
    // Input(name) maps to the same Int symbol across the assertion set,
    // and the same Value(vid) maps to the same Int symbol too).
    let mut input_seen: HashMap<String, Int> = HashMap::new();
    let mut value_seen: HashMap<ValueId, Int> = HashMap::new();

    for f in facts {
        collect_vars(f, &mut input_seen, &mut value_seen);
    }
    for p in paths {
        collect_vars(p, &mut input_seen, &mut value_seen);
    }
    collect_vars(term, &mut input_seen, &mut value_seen);

    let mut subst = Substitution::new();
    for (name, int) in &input_seen {
        subst = subst.with_input(name.clone(), int.clone());
    }
    for (vid, int) in &value_seen {
        subst = subst.with_value_id(*vid, int.clone());
    }

    let solver = z3::Solver::new();
    {
        let mut params = z3::Params::new();
        params.set_u32("timeout", timeout_ms.min(u32::MAX as u64) as u32);
        solver.set_params(&params);
    }

    for f in facts.iter().chain(paths.iter()) {
        match formula::lower_bool(f, &subst) {
            Ok(out) => {
                solver.assert(&out.term);
                for (_, meta) in &out.meta_fact_sets {
                    for m in meta {
                        solver.assert(m);
                    }
                }
            }
            Err(_) => {
                // Drop facts that don't lower (malformed templates, unbound
                // formals, etc.). They contribute no info; the prove still
                // proceeds against the remaining facts.
                continue;
            }
        }
    }

    let lowered_term = match formula::lower_bool(term, &subst) {
        Ok(out) => out.term,
        Err(_) => {
            // Query term itself can't be lowered — soundly Unknown.
            return None;
        }
    };

    let target: Bool = if negate { lowered_term.not() } else { lowered_term };
    solver.assert(&target);

    match solver.check() {
        z3::SatResult::Unsat => Some(true),
        z3::SatResult::Sat => Some(false),
        z3::SatResult::Unknown => None,
    }
}

/// Walk `term` and record every distinct `Input(name)` / `Value(vid)` leaf,
/// minting a fresh Z3 `Int` for each. Idempotent: a name/value_id seen
/// twice reuses the existing entry.
fn collect_vars(
    term: &ContractTerm,
    inputs: &mut HashMap<String, Int>,
    values: &mut HashMap<ValueId, Int>,
) {
    match term {
        ContractTerm::Var(ContractVar::Input(name)) => {
            inputs
                .entry(name.clone())
                .or_insert_with(|| Int::fresh_const(&format!("prove_in_{name}_")));
        }
        ContractTerm::Var(ContractVar::Value(vid)) => {
            values
                .entry(*vid)
                .or_insert_with(|| Int::fresh_const(&format!("prove_v{}_", vid.0)));
        }
        ContractTerm::Var(ContractVar::Output)
        | ContractTerm::Var(ContractVar::Formal(_))
        | ContractTerm::LitInt(_)
        | ContractTerm::LitFloat(_)
        | ContractTerm::LitBool(_) => {}
        ContractTerm::Arith { lhs, rhs, .. } | ContractTerm::Cmp { lhs, rhs, .. } => {
            collect_vars(lhs, inputs, values);
            collect_vars(rhs, inputs, values);
        }
        ContractTerm::BoolComb { operands, .. } => {
            for o in operands {
                collect_vars(o, inputs, values);
            }
        }
        ContractTerm::Not(inner) => collect_vars(inner, inputs, values),
        ContractTerm::PredicateApp { args, .. } => {
            for a in args {
                collect_vars(a, inputs, values);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    fn input(n: &str) -> ContractTerm {
        ContractTerm::Var(ContractVar::Input(n.to_string()))
    }
    fn vref(p: u64) -> ContractTerm {
        ContractTerm::Var(ContractVar::Value(ValueId(p)))
    }
    fn lit(n: i64) -> ContractTerm {
        ContractTerm::LitInt(n)
    }
    fn cmp(op: CmpOp, l: ContractTerm, r: ContractTerm) -> ContractTerm {
        ContractTerm::Cmp {
            op,
            lhs: Box::new(l),
            rhs: Box::new(r),
        }
    }

    #[test]
    fn prove_returns_proved_for_directly_known_fact() {
        // Facts: { k >= 0 }; Term: k >= 0. Trivially proved.
        let facts = vec![cmp(CmpOp::Ge, input("k"), lit(0))];
        let term = cmp(CmpOp::Ge, input("k"), lit(0));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Proved);
    }

    #[test]
    fn prove_returns_proved_for_entailed_term() {
        // Facts: { k >= 0 }; Term: k >= -1. Entailment, not literal match.
        let facts = vec![cmp(CmpOp::Ge, input("k"), lit(0))];
        let term = cmp(CmpOp::Ge, input("k"), lit(-1));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Proved);
    }

    #[test]
    fn prove_returns_proved_with_chained_bounds() {
        // Facts: { k >= 0, k <= 16 }; Term: k < 100. Proved via upper bound.
        let facts = vec![
            cmp(CmpOp::Ge, input("k"), lit(0)),
            cmp(CmpOp::Le, input("k"), lit(16)),
        ];
        let term = cmp(CmpOp::Lt, input("k"), lit(100));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Proved);
    }

    #[test]
    fn prove_returns_disproved_for_contradicted_term() {
        // Facts: { k >= 0 }; Term: k < 0. Term contradicts facts.
        let facts = vec![cmp(CmpOp::Ge, input("k"), lit(0))];
        let term = cmp(CmpOp::Lt, input("k"), lit(0));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Disproved);
    }

    #[test]
    fn prove_returns_unknown_when_no_info() {
        // Facts: empty; Term: k >= 0. Indeterminate.
        let facts: Vec<Fact> = vec![];
        let term = cmp(CmpOp::Ge, input("k"), lit(0));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn prove_uses_value_facts() {
        // Facts about a specific Value: { v(42) >= 0 }; Term: v(42) >= -1.
        let facts = vec![cmp(CmpOp::Ge, vref(42), lit(0))];
        let term = cmp(CmpOp::Ge, vref(42), lit(-1));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Proved);
    }

    #[test]
    fn prove_distinct_values_are_unrelated() {
        let facts = vec![cmp(CmpOp::Ge, vref(1), lit(0))];
        let term = cmp(CmpOp::Ge, vref(2), lit(0));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn prove_via_irbuilder_reads_factstack() {
        // End-to-end: drive prove through the IRBuilder's FactStack.
        // compiler.value-id-and-fact-leaves: facts are value_id-anchored.
        use crate::circuit_input::InputPath;
        let mut b = IRBuilder::new();
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();
        b.facts.insert_for(
            k_vid,
            cmp(
                CmpOp::Ge,
                ContractTerm::Var(ContractVar::Value(k_vid)),
                lit(0),
            ),
        );
        let term = cmp(
            CmpOp::Ge,
            ContractTerm::Var(ContractVar::Value(k_vid)),
            lit(-5),
        );
        assert_eq!(prove(&b, &term), ProveOutcome::Proved);
    }
}
