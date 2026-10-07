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

use std::collections::{BTreeSet, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use z3::ast::{Bool, Int, Real};

use crate::builder::IRBuilder;

// Per-process counters for prove-outcome and SMT-path telemetry. Used
// by the DS-1000 acceptance/rejection sweep to characterize where
// rejection comes from (compile_time prove failure vs. lenient
// witness emit vs. plain Unknown).
pub static PROVE_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
pub static PROVE_PROVED_COUNT: AtomicU64 = AtomicU64::new(0);
pub static PROVE_DISPROVED_COUNT: AtomicU64 = AtomicU64::new(0);
pub static PROVE_UNKNOWN_COUNT: AtomicU64 = AtomicU64::new(0);
pub static PROVE_Z3_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
pub static PROVE_FASTPATH_HITS: AtomicU64 = AtomicU64::new(0);

pub fn reset_prove_counters() {
    PROVE_CALL_COUNT.store(0, Ordering::Relaxed);
    PROVE_PROVED_COUNT.store(0, Ordering::Relaxed);
    PROVE_DISPROVED_COUNT.store(0, Ordering::Relaxed);
    PROVE_UNKNOWN_COUNT.store(0, Ordering::Relaxed);
    PROVE_Z3_INVOCATIONS.store(0, Ordering::Relaxed);
    PROVE_FASTPATH_HITS.store(0, Ordering::Relaxed);
}
use crate::optim::predicates::facts::{Fact, FactScope};
use crate::optim::predicates::formula::{self, ContractTerm, ContractVar, Substitution};
use crate::optim::z3_budget;
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
    /// No discharge succeeded — could mean an exhausted step budget, no
    /// entailment, or no contradiction. Treat as "no information."
    Unknown,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Query whether `term` follows from the visible facts + path conditions
/// currently held by `b.facts`. See module docs for outcome semantics.
///
/// Reads `ZINNIA_SMT_PROVE_RLIMIT` for the per-check budget in Z3 steps;
/// defaults to `z3_budget::PROVE_RLIMIT` (a step budget, not a timeout, so
/// the outcome does not depend on machine load). The same budget applies to
/// both the entailment check and (if needed) the contradiction check.
/// Queries that exhaust it surface as `ProveOutcome::Unknown` (which callers
/// treat as "no information" — same as if the prove had been skipped).
pub fn prove(b: &IRBuilder, term: &ContractTerm) -> ProveOutcome {
    // A/B-harness kill switch: under `ZINNIA_REQ_DISABLE=1` the prove
    // layer returns Unknown unconditionally. The discharge_requires
    // lenient branch then emits the witness check (`IR::Assert`), so
    // preconditions are still enforced at proof time — the soundness
    // floor is intact. See `compiler.verification-ab-disable-harness`.
    if crate::optim::resolver::req_disabled() {
        return ProveOutcome::Unknown;
    }
    let facts: Vec<&Fact> = b.facts.visible_facts().into_iter().collect();
    let paths: Vec<&Fact> = b.facts.visible_path_conditions().into_iter().collect();
    prove_typed(term, &facts, &paths, Floats(Some(b)))
}

/// Where to learn which values are floats: the IR of the builder the facts
/// come from, when there is one. Asked only for the facts of a query that
/// reaches Z3.
#[derive(Clone, Copy)]
struct Floats<'a>(Option<&'a IRBuilder>);

impl Floats<'_> {
    /// The values among `terms`' variables that the IR computes as floats.
    fn among(self, terms: &[&ContractTerm]) -> BTreeSet<ValueId> {
        let Some(b) = self.0 else { return BTreeSet::new() };
        let mut values = BTreeSet::new();
        for term in terms {
            fact_walk_vars(term, &mut |var| {
                if let ContractVar::Value(vid) = *var {
                    if b.value_id_to_stmt_id(vid).is_some_and(|sid| b.is_float_stmt(sid)) {
                        values.insert(vid);
                    }
                }
            });
        }
        values
    }
}

/// Same as [`prove`] but takes an explicit fact list and path-condition
/// list. Useful for tests and for deferred queries against a stashed
/// [`FactScope`]. Which values are floats is inferred from the terms alone
/// (see [`substitution_for`]).
pub fn prove_with_facts(term: &ContractTerm, facts: &[Fact], paths: &[Fact]) -> ProveOutcome {
    let facts: Vec<&Fact> = facts.iter().collect();
    let paths: Vec<&Fact> = paths.iter().collect();
    prove_typed(term, &facts, &paths, Floats(None))
}

/// [`prove_with_facts`] over borrowed facts, with where to learn floats.
fn prove_typed(term: &ContractTerm, facts: &[&Fact], paths: &[&Fact], floats: Floats) -> ProveOutcome {
    PROVE_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    let outcome = prove_with_facts_inner(term, facts, paths, floats);
    match outcome {
        ProveOutcome::Proved => PROVE_PROVED_COUNT.fetch_add(1, Ordering::Relaxed),
        ProveOutcome::Disproved => PROVE_DISPROVED_COUNT.fetch_add(1, Ordering::Relaxed),
        ProveOutcome::Unknown => PROVE_UNKNOWN_COUNT.fetch_add(1, Ordering::Relaxed),
    };
    outcome
}

fn prove_with_facts_inner(term: &ContractTerm, facts: &[&Fact], paths: &[&Fact], floats: Floats) -> ProveOutcome {
    // Compound-term decomposition. For And/Or/Not we recurse on
    // operands and combine outcomes structurally:
    //
    //   And: any Disproved → Disproved; else all Proved → Proved; else Unknown.
    //   Or : any Proved   → Proved;    else all Disproved → Disproved; else Unknown.
    //   Not: invert.
    //
    // Why decompose: per-atom proves let the connectivity fast-path
    // fire independently on each leaf, which is the only way it can
    // help compound queries like `And(idx >= 0, idx < hi)` (the second
    // leaf has disconnected vars and is fact-bare). Without
    // decomposition, the whole-term connectivity check trips on the OR
    // case (e.g. `pow_i`'s `base != 0 OR exp >= 0` where each disjunct
    // is independently Disprovable but the term-level vars are
    // disconnected), incorrectly returning Unknown where Z3 returns
    // Disproved.
    //
    // Soundness: combining per-atom outcomes the standard Boolean way
    // is sound iff each atomic prove is sound. The atomic path either
    // short-circuits (Unknown — always safe) or invokes Z3 with the
    // full fact context.
    match term {
        ContractTerm::LitBool(b) => {
            return if *b {
                ProveOutcome::Proved
            } else {
                ProveOutcome::Disproved
            };
        }
        ContractTerm::BoolComb { op, operands } => {
            use crate::optim::predicates::formula::BoolOp;
            let mut any_proved = false;
            let mut any_disproved = false;
            let mut all_proved = true;
            let mut all_disproved = true;
            for sub in operands {
                match prove_typed(sub, facts, paths, floats) {
                    ProveOutcome::Proved => {
                        any_proved = true;
                        all_disproved = false;
                    }
                    ProveOutcome::Disproved => {
                        any_disproved = true;
                        all_proved = false;
                    }
                    ProveOutcome::Unknown => {
                        all_proved = false;
                        all_disproved = false;
                    }
                }
            }
            return match op {
                BoolOp::And => {
                    if any_disproved {
                        ProveOutcome::Disproved
                    } else if all_proved {
                        ProveOutcome::Proved
                    } else {
                        ProveOutcome::Unknown
                    }
                }
                BoolOp::Or => {
                    if any_proved {
                        ProveOutcome::Proved
                    } else if all_disproved {
                        ProveOutcome::Disproved
                    } else {
                        ProveOutcome::Unknown
                    }
                }
            };
        }
        ContractTerm::Not(inner) => {
            return match prove_typed(inner, facts, paths, floats) {
                ProveOutcome::Proved => ProveOutcome::Disproved,
                ProveOutcome::Disproved => ProveOutcome::Proved,
                ProveOutcome::Unknown => ProveOutcome::Unknown,
            };
        }
        _ => {}
    }

    // Atomic path — single Cmp / PredicateApp / Var / etc.
    prove_atomic_with_facts(term, facts, paths, floats)
}

/// Atomic-term prove. Runs the two pre-Z3 fast-paths first, then falls
/// through to the Z3 solver. Callers go through [`prove_with_facts`],
/// which decomposes compound terms and only reaches this for leaves.
fn prove_atomic_with_facts(term: &ContractTerm, facts: &[&Fact], paths: &[&Fact], floats: Floats) -> ProveOutcome {
    // Pre-Z3 fast-path 1: pure-opaque PredicateApp. If `term` is a
    // PredicateApp whose registered kind has no numeric meta-facts
    // (e.g. `is_sorted`), Z3 can only entail or contradict it via a
    // fact mentioning the same `(cache_key, args)` symbol. If no fact
    // does, both sat-checks would trivially find a model — Unknown.
    if let Some(atoms) = collect_pure_opaque_atoms(term) {
        let mentioned = facts
            .iter()
            .chain(paths.iter())
            .any(|f| any_predicate_app_matches(f, &atoms));
        if !mentioned {
            PROVE_FASTPATH_HITS.fetch_add(1, Ordering::Relaxed);
            return ProveOutcome::Unknown;
        }
    }

    // Pre-Z3 fast-path 2: numeric connectivity. Build union-find over
    // the fact graph (facts as hyperedges over the vars they mention).
    // For the atomic to be provable, its vars must (a) all sit in the
    // same fact-graph component, and (b) that component must be
    // "touched" by at least one fact's mention. Otherwise Z3 has no
    // chain to constrain the atom — both sat-checks are sat → Unknown.
    //
    // This is load-bearing for the hot DynamicNDArray `result[i] = ...`
    // path: each per-iteration `i < hi` discharge query names freshly
    // minted SSA wires that no fact relates. Decomposition + this
    // check lets us skip the Z3 call entirely on those.
    //
    // The query is then asked over the facts of its component only: a fact
    // sharing no variable with it cannot change whether it is entailed or
    // contradicted (unless the other facts contradict each other, which is
    // no reason to conclude anything about the query). Fewer facts keep
    // each Z3 query small however many unrelated facts a program has.
    let mut relevant: Option<(Vec<&Fact>, Vec<&Fact>)> = None;
    if let Some(refs) = collect_concrete_var_refs(term) {
        if !refs.is_empty() {
            let mut uf = UnionFind::default();
            for v in &refs {
                uf.find(v);
            }
            let mut fact_buf: Vec<ContractVar> = Vec::new();
            let mut all_fact_vars: std::collections::HashSet<ContractVar> =
                std::collections::HashSet::new();
            for f in facts.iter().chain(paths.iter()) {
                fact_buf.clear();
                fact_walk_vars(f, &mut |v| fact_buf.push(v.clone()));
                if fact_buf.is_empty() {
                    continue;
                }
                for v in &fact_buf {
                    all_fact_vars.insert(v.clone());
                }
                let head = fact_buf[0].clone();
                for v in fact_buf.iter().skip(1) {
                    uf.union(&head, v);
                }
            }
            let query_roots: std::collections::HashSet<ContractVar> =
                refs.iter().map(|v| uf.find(v)).collect();
            let touched_roots: std::collections::HashSet<ContractVar> =
                all_fact_vars.iter().map(|v| uf.find(v)).collect();
            let same_component = query_roots.len() == 1;
            let touched = query_roots.iter().any(|r| touched_roots.contains(r));
            if !same_component || !touched {
                PROVE_FASTPATH_HITS.fetch_add(1, Ordering::Relaxed);
                return ProveOutcome::Unknown;
            }
            let mut in_component = |f: &Fact| {
                let mut hit = false;
                fact_walk_vars(f, &mut |v| hit |= query_roots.contains(&uf.find(v)));
                hit
            };
            relevant = Some((
                facts.iter().copied().filter(|f| in_component(f)).collect(),
                paths.iter().copied().filter(|f| in_component(f)).collect(),
            ));
        }
    }
    let (facts, paths) = match &relevant {
        Some((facts, paths)) => (facts.as_slice(), paths.as_slice()),
        None => (facts, paths),
    };

    let rlimit = std::env::var("ZINNIA_SMT_PROVE_RLIMIT")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(z3_budget::PROVE_RLIMIT);

    let entails = check_unsat(facts, paths, term, floats, /* negate */ true, rlimit);
    match entails {
        Some(true) => return ProveOutcome::Proved,
        Some(false) | None => {}
    }
    let contradicts = check_unsat(facts, paths, term, floats, /* negate */ false, rlimit);
    match contradicts {
        Some(true) => ProveOutcome::Disproved,
        _ => ProveOutcome::Unknown,
    }
}

/// Walk `term`. Succeeds (returns `Some(atoms)`) iff every leaf is a
/// `PredicateApp` whose registered kind is pure-opaque (no numeric
/// meta-facts), wrapped in any combination of `Not` / `BoolComb` /
/// `LitBool`. Returns `None` the moment a numeric leaf, comparison,
/// arithmetic node, or non-pure-opaque predicate appears — those can
/// interact with arithmetic facts in non-obvious ways, so Z3 must
/// stay in the loop.
///
/// Atoms are keyed by `(cache_key, args)`, NOT `(kind, args)`. The
/// registry's `cache_key` collapses kind aliases that share a Z3
/// symbol (e.g. `is_monotone_nondecreasing → "is_sorted"`), so the
/// fact-matching step downstream correctly treats those as
/// interchangeable.
fn collect_pure_opaque_atoms(
    term: &ContractTerm,
) -> Option<Vec<(&'static str, Vec<ContractTerm>)>> {
    let registry = crate::optim::predicates::registry::registry();
    let mut atoms = Vec::new();
    if !walk_pure_opaque(term, registry, &mut atoms) {
        return None;
    }
    Some(atoms)
}

fn walk_pure_opaque(
    term: &ContractTerm,
    registry: &std::collections::HashMap<
        &'static str,
        crate::optim::predicates::registry::PredicateRegistration,
    >,
    atoms: &mut Vec<(&'static str, Vec<ContractTerm>)>,
) -> bool {
    match term {
        ContractTerm::PredicateApp { kind, args } => {
            let reg = match registry.get(kind.as_str()) {
                Some(r) => r,
                None => return false,
            };
            if !reg.is_pure_opaque {
                return false;
            }
            atoms.push((reg.cache_key, args.clone()));
            true
        }
        ContractTerm::Not(inner) => walk_pure_opaque(inner, registry, atoms),
        ContractTerm::BoolComb { operands, .. } => operands
            .iter()
            .all(|op| walk_pure_opaque(op, registry, atoms)),
        ContractTerm::LitBool(_) => true,
        // Anything that can carry numeric content interacts with
        // arithmetic facts.
        _ => false,
    }
}

/// Walk a fact looking for any `PredicateApp` whose `(cache_key, args)`
/// matches one of `atoms`. Compares args structurally (`ContractTerm`
/// derives `Eq`). Resolves the fact's `kind` to its `cache_key` via
/// the registry so aliased symbols match correctly.
fn any_predicate_app_matches(
    fact: &ContractTerm,
    atoms: &[(&'static str, Vec<ContractTerm>)],
) -> bool {
    let registry = crate::optim::predicates::registry::registry();
    fn rec(
        term: &ContractTerm,
        atoms: &[(&'static str, Vec<ContractTerm>)],
        registry: &std::collections::HashMap<
            &'static str,
            crate::optim::predicates::registry::PredicateRegistration,
        >,
    ) -> bool {
        match term {
            ContractTerm::PredicateApp { kind, args } => {
                let fact_key = registry
                    .get(kind.as_str())
                    .map(|r| r.cache_key)
                    .unwrap_or(kind.as_str());
                atoms.iter().any(|(k, a)| *k == fact_key && a == args)
            }
            ContractTerm::Not(inner) => rec(inner, atoms, registry),
            ContractTerm::BoolComb { operands, .. } => {
                operands.iter().any(|op| rec(op, atoms, registry))
            }
            ContractTerm::Cmp { lhs, rhs, .. } | ContractTerm::Arith { lhs, rhs, .. } => {
                rec(lhs, atoms, registry) || rec(rhs, atoms, registry)
            }
            _ => false,
        }
    }
    rec(fact, atoms, registry)
}

/// Collect every concrete `Var(Input | Value | ShapeParam)` reference
/// in `term`. Returns `None` if the term contains a template-only
/// placeholder (`Output` or `Formal`) — those should be substituted
/// before reaching `prove`, but defensively falling through to Z3 is
/// safer than guessing.
fn collect_concrete_var_refs(
    term: &ContractTerm,
) -> Option<std::collections::HashSet<ContractVar>> {
    let mut out = std::collections::HashSet::new();
    if !walk_concrete_vars(term, &mut out) {
        return None;
    }
    Some(out)
}

fn walk_concrete_vars(
    term: &ContractTerm,
    out: &mut std::collections::HashSet<ContractVar>,
) -> bool {
    match term {
        ContractTerm::Var(v) => match v {
            ContractVar::Output | ContractVar::Formal(_) => false,
            other => {
                out.insert(other.clone());
                true
            }
        },
        ContractTerm::Cmp { lhs, rhs, .. } | ContractTerm::Arith { lhs, rhs, .. } => {
            walk_concrete_vars(lhs, out) && walk_concrete_vars(rhs, out)
        }
        ContractTerm::BoolComb { operands, .. } => {
            operands.iter().all(|op| walk_concrete_vars(op, out))
        }
        ContractTerm::Not(inner) => walk_concrete_vars(inner, out),
        ContractTerm::PredicateApp { args, .. } => args.iter().all(|a| walk_concrete_vars(a, out)),
        ContractTerm::LitInt(_) | ContractTerm::LitFloat(_) | ContractTerm::LitBool(_) => true,
    }
}

/// Walk a `ContractTerm` and call `cb` with every concrete
/// `Var(Input | Value | ShapeParam)` leaf. Template placeholders
/// (`Output`, `Formal`) are skipped — they don't bind anything at
/// query time.
fn fact_walk_vars(term: &ContractTerm, cb: &mut impl FnMut(&ContractVar)) {
    match term {
        ContractTerm::Var(v) => match v {
            ContractVar::Output | ContractVar::Formal(_) => {}
            _ => cb(v),
        },
        ContractTerm::Cmp { lhs, rhs, .. } | ContractTerm::Arith { lhs, rhs, .. } => {
            fact_walk_vars(lhs, cb);
            fact_walk_vars(rhs, cb);
        }
        ContractTerm::BoolComb { operands, .. } => {
            for op in operands {
                fact_walk_vars(op, cb);
            }
        }
        ContractTerm::Not(inner) => fact_walk_vars(inner, cb),
        ContractTerm::PredicateApp { args, .. } => {
            for arg in args {
                fact_walk_vars(arg, cb);
            }
        }
        ContractTerm::LitInt(_) | ContractTerm::LitFloat(_) | ContractTerm::LitBool(_) => {}
    }
}

/// Minimal union-find over `ContractVar` for the fast-path's fact-graph
/// connectivity check. Path compression on `find`; no rank balancing
/// (the working set is small and the cost is dominated by Z3 if we
/// don't short-circuit anyway).
#[derive(Default)]
struct UnionFind {
    parent: std::collections::HashMap<ContractVar, ContractVar>,
}

impl UnionFind {
    fn find(&mut self, v: &ContractVar) -> ContractVar {
        let parent_of = match self.parent.get(v) {
            Some(p) => p.clone(),
            None => {
                self.parent.insert(v.clone(), v.clone());
                return v.clone();
            }
        };
        if &parent_of == v {
            return parent_of;
        }
        let root = self.find(&parent_of);
        self.parent.insert(v.clone(), root.clone());
        root
    }

    fn union(&mut self, a: &ContractVar, b: &ContractVar) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(ra, rb);
        }
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

pub fn minimize_shape_param(
    symbol: &str,
    facts: &[Fact],
    lower_bound: i64,
    rlimit: u32,
) -> Option<i64> {
    optimize_shape_param(symbol, facts, lower_bound, false, rlimit)
}

pub fn maximize_shape_param(
    symbol: &str,
    facts: &[Fact],
    lower_bound: i64,
    rlimit: u32,
) -> Option<i64> {
    let max = optimize_shape_param(symbol, facts, lower_bound, true, rlimit)?;
    let upper_bound = ContractTerm::le(
        ContractTerm::Var(ContractVar::ShapeParam(symbol.to_string())),
        ContractTerm::lit(max),
    );
    match prove_with_facts(&upper_bound, facts, &[]) {
        ProveOutcome::Proved => Some(max),
        ProveOutcome::Disproved | ProveOutcome::Unknown => None,
    }
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
/// if sat, `None` if Z3 returned Unknown (typically the step budget ran out).
fn check_unsat(
    facts: &[&Fact],
    paths: &[&Fact],
    term: &ContractTerm,
    floats: Floats,
    negate: bool,
    rlimit: u32,
) -> Option<bool> {
    PROVE_Z3_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    // One symbol per Input / Value(vid) / shape parameter the assembled
    // formula references, shared by every assertion.
    let terms: Vec<&ContractTerm> = facts.iter().chain(paths).copied().chain(std::iter::once(term)).collect();
    let subst = substitution_for(&terms, &floats.among(&terms));

    let solver = z3_budget::solver(rlimit);

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

    let target: Bool = if negate {
        lowered_term.not()
    } else {
        lowered_term
    };
    solver.assert(&target);

    match z3_budget::check("prove.entails", &solver) {
        z3::SatResult::Unsat => Some(true),
        z3::SatResult::Sat => Some(false),
        z3::SatResult::Unknown => None,
    }
}

fn optimize_shape_param(
    symbol: &str,
    facts: &[Fact],
    lower_bound: i64,
    maximize: bool,
    rlimit: u32,
) -> Option<i64> {
    let symbol_term = ContractTerm::Var(ContractVar::ShapeParam(symbol.to_string()));
    let terms: Vec<&ContractTerm> = facts.iter().chain(std::iter::once(&symbol_term)).collect();
    let subst = substitution_for(&terms, &BTreeSet::new());

    let opt = z3_budget::optimize(rlimit);
    for f in facts {
        let out = formula::lower_bool(f, &subst).ok()?;
        opt.assert(&out.term);
        for (_, meta) in &out.meta_fact_sets {
            for m in meta {
                opt.assert(m);
            }
        }
    }
    let target = match subst.shape_param(symbol)? {
        crate::optim::smt_encoding::Z3Term::Int(i) => i.clone(),
        _ => return None,
    };
    opt.assert(&target.ge(Int::from_i64(lower_bound)));
    if maximize {
        opt.maximize(&target);
    } else {
        opt.minimize(&target);
    }
    match z3_budget::check_optimize("prove.shape_param", &opt) {
        z3::SatResult::Sat => opt
            .get_model()
            .and_then(|model| model.eval(&target, true))
            .and_then(|v| v.as_i64()),
        z3::SatResult::Unsat | z3::SatResult::Unknown => None,
    }
}

/// Z3 symbols for every variable of `terms`, minted in order of first
/// appearance: a `Real` for a float, an `Int` otherwise; shape parameters
/// are `Int`s. The floats are `floats` and, to a fixpoint, every variable
/// compared or combined in arithmetic with a float literal or another real
/// variable. A float is a point of the Q32 grid, not an integer: as an
/// `Int`, the fact `x > -0.5` would entail `x >= 0` (and `-0.5 < x < 0.5`
/// would entail `x == 0`). Modelling an integer as a real only forgets its
/// integrality, which is sound.
fn substitution_for(terms: &[&ContractTerm], floats: &BTreeSet<ValueId>) -> Substitution {
    let mut reals: HashSet<ContractVar> = floats.iter().map(|vid| ContractVar::Value(*vid)).collect();
    loop {
        let before = reals.len();
        for term in terms {
            mark_reals(term, &mut reals);
        }
        if reals.len() == before {
            break;
        }
    }
    let mut subst = Substitution::new();
    let mut seen: HashSet<ContractVar> = HashSet::new();
    for term in terms {
        fact_walk_vars(term, &mut |var| {
            if !seen.insert(var.clone()) {
                return;
            }
            subst = bind(std::mem::take(&mut subst), var, reals.contains(var));
        });
    }
    subst
}

/// `subst` with a fresh symbol for `var`: a `Real` when `real`, else an `Int`.
fn bind(subst: Substitution, var: &ContractVar, real: bool) -> Substitution {
    use crate::optim::smt_encoding::Z3Term;
    let fresh = |name: String| if real { Z3Term::Real(Real::fresh_const(&name)) } else { Z3Term::Int(Int::fresh_const(&name)) };
    match var {
        ContractVar::Input(name) => subst.with_input_term(name.clone(), fresh(format!("prove_in_{name}_"))),
        ContractVar::Value(vid) => subst.with_value_id_term(*vid, fresh(format!("prove_v{}_", vid.0))),
        ContractVar::ShapeParam(name) => subst.with_shape_param(name.clone(), Int::fresh_const(&format!("prove_shape_{name}_"))),
        ContractVar::Output | ContractVar::Formal(_) => subst,
    }
}

/// Mark as real every variable of a comparison or an arithmetic term one of
/// whose sides mentions a float literal or a real variable.
fn mark_reals(term: &ContractTerm, reals: &mut HashSet<ContractVar>) {
    match term {
        ContractTerm::Arith { lhs, rhs, .. } | ContractTerm::Cmp { lhs, rhs, .. } => {
            if mentions_real(lhs, reals) || mentions_real(rhs, reals) {
                for side in [lhs, rhs] {
                    fact_walk_vars(side, &mut |var| {
                        if !matches!(var, ContractVar::ShapeParam(_)) {
                            reals.insert(var.clone());
                        }
                    });
                }
            }
            mark_reals(lhs, reals);
            mark_reals(rhs, reals);
        }
        ContractTerm::BoolComb { operands, .. } => operands.iter().for_each(|o| mark_reals(o, reals)),
        ContractTerm::Not(inner) => mark_reals(inner, reals),
        _ => {}
    }
}

fn mentions_real(term: &ContractTerm, reals: &HashSet<ContractVar>) -> bool {
    match term {
        ContractTerm::LitFloat(_) => true,
        ContractTerm::Var(var) => reals.contains(var),
        ContractTerm::Arith { lhs, rhs, .. } | ContractTerm::Cmp { lhs, rhs, .. } => {
            mentions_real(lhs, reals) || mentions_real(rhs, reals)
        }
        _ => false,
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
    fn shape(n: &str) -> ContractTerm {
        ContractTerm::Var(ContractVar::ShapeParam(n.to_string()))
    }
    fn cmp(op: CmpOp, l: ContractTerm, r: ContractTerm) -> ContractTerm {
        ContractTerm::Cmp {
            op,
            lhs: Box::new(l),
            rhs: Box::new(r),
        }
    }

    /// The query is asked over the facts of its own component: facts about
    /// other values, even contradictory ones, prove nothing about it.
    #[test]
    fn unrelated_contradictory_facts_prove_nothing() {
        let facts = vec![
            cmp(CmpOp::Ge, vref(1), lit(0)),
            cmp(CmpOp::Gt, vref(2), lit(1)),
            cmp(CmpOp::Lt, vref(2), lit(0)),
        ];
        assert_eq!(prove_with_facts(&cmp(CmpOp::Gt, vref(1), lit(5)), &facts, &[]), ProveOutcome::Unknown);
        assert_eq!(prove_with_facts(&cmp(CmpOp::Ge, vref(1), lit(-1)), &facts, &[]), ProveOutcome::Proved);
    }

    /// A float is a real: `x > -0.5` does not entail `x >= 0`, and
    /// `-0.5 < x < 0.5` does not entail `x == 0`.
    #[test]
    fn float_facts_are_over_the_reals() {
        let half = || ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(0.5));
        let minus_half = || ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(-0.5));
        let zero = || ContractTerm::LitFloat(crate::optim::predicates::formula::ContractFloat(0.0));
        let facts = vec![cmp(CmpOp::Gt, vref(1), minus_half()), cmp(CmpOp::Lt, vref(1), half())];
        assert_eq!(prove_with_facts(&cmp(CmpOp::Ge, vref(1), zero()), &facts, &[]), ProveOutcome::Unknown);
        assert_eq!(prove_with_facts(&cmp(CmpOp::Ne, vref(1), zero()), &facts, &[]), ProveOutcome::Unknown);
        assert_eq!(prove_with_facts(&cmp(CmpOp::Lt, vref(1), lit(1)), &facts, &[]), ProveOutcome::Proved);
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
    fn shape_param_optimizer_extracts_bounds_beyond_old_scan_window() {
        let facts = vec![
            cmp(CmpOp::Ge, shape("N"), lit(1)),
            cmp(CmpOp::Le, shape("N"), lit(5000)),
        ];
        assert_eq!(minimize_shape_param("N", &facts, 1, z3_budget::RESOLVER_RLIMIT), Some(1));
        assert_eq!(maximize_shape_param("N", &facts, 1, z3_budget::RESOLVER_RLIMIT), Some(5000));
    }

    #[test]
    fn shape_param_optimizer_rejects_unbounded_upper_objective() {
        let facts = vec![cmp(CmpOp::Ge, shape("N"), lit(1))];
        assert_eq!(maximize_shape_param("N", &facts, 1, z3_budget::RESOLVER_RLIMIT), None);
    }

    #[test]
    fn prove_returns_disproved_for_contradicted_term() {
        // Facts: { k >= 0 }; Term: k < 0. Term contradicts facts.
        let facts = vec![cmp(CmpOp::Ge, input("k"), lit(0))];
        let term = cmp(CmpOp::Lt, input("k"), lit(0));
        assert_eq!(
            prove_with_facts(&term, &facts, &[]),
            ProveOutcome::Disproved
        );
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

    #[test]
    fn prove_consumes_statement_branch_path_condition() {
        use crate::circuit_input::InputPath;
        use crate::optim::predicates::ScopeKind;

        let mut b = IRBuilder::new();
        let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
        let zero = b.ir_constant_int(0);
        let cond = b.ir_greater_than_i(&x, &zero);
        let cond_vid = cond.value_id().unwrap();
        let path = cmp(CmpOp::Eq, vref(cond_vid.0), lit(1));

        b.facts.enter(ScopeKind::Conditional, Some(path.clone()));
        assert_eq!(prove(&b, &path), ProveOutcome::Proved);
        let _ = b.facts.leave();
    }

    #[test]
    fn prove_uses_guarded_join_fact_only_when_guard_visible() {
        use crate::optim::predicates::{FactStack, ScopeKind};

        let cond = cmp(CmpOp::Eq, vref(1), lit(1));
        let not_cond = cmp(CmpOp::Eq, vref(1), lit(0));
        let then_fact = cmp(CmpOp::Ge, vref(10), lit(0));
        let else_fact = cmp(CmpOp::Le, vref(10), lit(-1));
        let mut stk = FactStack::new();

        stk.enter(ScopeKind::Conditional, Some(cond.clone()));
        stk.insert_for(ValueId(10), then_fact.clone());
        let then_scope = stk.leave();

        stk.enter(ScopeKind::Conditional, Some(not_cond.clone()));
        stk.insert_for(ValueId(10), else_fact);
        let else_scope = stk.leave();

        stk.merge_branches(then_scope, else_scope);
        let joined_facts: Vec<Fact> = stk.visible_facts().into_iter().cloned().collect();

        assert_eq!(
            prove_with_facts(&then_fact, &joined_facts, &[]),
            ProveOutcome::Unknown,
            "post-join guarded fact must stay conditional without a visible guard"
        );
        assert_eq!(
            prove_with_facts(&then_fact, &joined_facts, &[cond]),
            ProveOutcome::Proved,
            "visible branch guard should discharge the guarded implication"
        );
    }

    // ── Pure-opaque fast-path (skip Z3 when no fact mentions the term)
    //
    // These tests pin down the soundness boundary of the
    // `collect_pure_opaque_atoms` fast-path:
    //   - empty facts + pure-opaque term ⇒ Unknown without Z3
    //   - irrelevant facts (different vid or different kind) ⇒ Unknown
    //   - matching fact ⇒ goes through Z3 (must still return Proved)
    //   - numeric / mixed term ⇒ falls through to Z3
    //   - non-pure-opaque predicate (`nnz`) ⇒ falls through to Z3

    fn pred(kind: &str, args: Vec<ContractTerm>) -> ContractTerm {
        ContractTerm::PredicateApp {
            kind: kind.to_string(),
            args,
        }
    }

    #[test]
    fn pure_opaque_no_facts_returns_unknown_fast_path() {
        let term = pred("is_sorted", vec![vref(1)]);
        assert_eq!(prove_with_facts(&term, &[], &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn pure_opaque_unrelated_facts_returns_unknown_fast_path() {
        // Numeric facts about different vids don't touch the opaque symbol.
        let facts = vec![
            cmp(CmpOp::Ge, vref(99), lit(0)),
            cmp(CmpOp::Le, vref(99), lit(10)),
            // Same kind but different argument vid — different cache key.
            pred("is_sorted", vec![vref(42)]),
        ];
        let term = pred("is_sorted", vec![vref(1)]);
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn pure_opaque_with_matching_fact_proves() {
        // A fact about the same (kind, args) is the entire point of the
        // predicate registry — the fast-path should NOT short-circuit
        // here; Z3 should be invoked and return Proved.
        let facts = vec![pred("is_sorted", vec![vref(1)])];
        let term = pred("is_sorted", vec![vref(1)]);
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Proved);
    }

    #[test]
    fn pure_opaque_combined_with_numeric_falls_through_to_z3() {
        // Term mixes a pure-opaque atom with a numeric comparison —
        // not a pure-opaque expression, so Z3 must still run.
        let facts = vec![cmp(CmpOp::Ge, input("k"), lit(0))];
        let term = ContractTerm::BoolComb {
            op: crate::optim::predicates::formula::BoolOp::And,
            operands: vec![
                pred("is_sorted", vec![vref(1)]),
                cmp(CmpOp::Ge, input("k"), lit(0)),
            ],
        };
        // Without a fact about is_sorted(vref(1)), can't be proved; but
        // also can't be short-circuited at the fast-path because the
        // term isn't purely opaque.
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn non_pure_opaque_predicate_does_not_take_fast_path() {
        // `nnz` carries a numeric pred-value (`nnz >= 0`); short-circuit
        // is unsound there. The fast-path must NOT engage. With a
        // non-empty fact list and no nnz-mentioning facts, Z3 still
        // gets called — outcome stays Unknown but goes through SMT.
        let facts = vec![cmp(CmpOp::Ge, input("k"), lit(0))];
        let term = pred("nnz", vec![vref(1)]);
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn negated_pure_opaque_no_facts_returns_unknown_fast_path() {
        // ¬is_sorted(vref(1)) is still purely opaque; same fast-path.
        let term = ContractTerm::Not(Box::new(pred("is_sorted", vec![vref(1)])));
        assert_eq!(prove_with_facts(&term, &[], &[]), ProveOutcome::Unknown);
    }

    // ── Numeric-Cmp fact-bareness fast-path
    //
    // Mirrors the discharge_index_below_value / discharge_index_in_range
    // call patterns that drove the F-shape compile-time blow-up before
    // this optimization: per-write Cmp queries over freshly-minted
    // SSA wires that no fact mentions.

    #[test]
    fn cmp_with_no_facts_mentioning_vid_returns_unknown_fast_path() {
        // Query: vref(42) >= 0. Facts: empty. Z3 should not be invoked.
        let term = cmp(CmpOp::Ge, vref(42), lit(0));
        assert_eq!(prove_with_facts(&term, &[], &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn cmp_with_facts_only_about_unrelated_vids_returns_unknown_fast_path() {
        // Facts talk about vref(1) and vref(2). Query asks about vref(99).
        // No transitive chain can possibly constrain vref(99).
        let facts = vec![
            cmp(CmpOp::Ge, vref(1), lit(0)),
            cmp(CmpOp::Le, vref(2), lit(100)),
        ];
        let term = cmp(CmpOp::Lt, vref(99), lit(50));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn cmp_chained_via_arith_eq_still_uses_z3() {
        // Facts: { vref(7) = vref(8) + 5, vref(8) >= 0 }. The first fact
        // mentions vref(7) so the fast-path does NOT short-circuit; Z3
        // sees the chain and proves vref(7) >= 5.
        let facts = vec![
            cmp(
                CmpOp::Eq,
                vref(7),
                ContractTerm::Arith {
                    op: crate::optim::predicates::formula::ArithOp::Add,
                    lhs: Box::new(vref(8)),
                    rhs: Box::new(lit(5)),
                },
            ),
            cmp(CmpOp::Ge, vref(8), lit(0)),
        ];
        let term = cmp(CmpOp::Ge, vref(7), lit(5));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Proved);
    }

    #[test]
    fn cmp_with_runtime_upper_bound_no_relating_fact_returns_unknown_fast_path() {
        // The exact shape `dyn_setitem` hits: prove `0 <= vref(10) < vref(20)`
        // with no fact relating the two. Should be Unknown without Z3.
        let term = ContractTerm::BoolComb {
            op: crate::optim::predicates::formula::BoolOp::And,
            operands: vec![
                cmp(CmpOp::Ge, vref(10), lit(0)),
                cmp(CmpOp::Lt, vref(10), vref(20)),
            ],
        };
        assert_eq!(prove_with_facts(&term, &[], &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn cmp_input_var_with_unrelated_facts_returns_unknown_fast_path() {
        // Inputs and ShapeParams participate too.
        let facts = vec![cmp(CmpOp::Ge, vref(1), lit(0))];
        let term = cmp(CmpOp::Ge, input("k"), lit(0));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Unknown);
    }

    #[test]
    fn literal_only_term_falls_through_to_z3() {
        // A literal-only term has no var refs, so the fast-path doesn't
        // fire (it's gated on `refs.is_empty()`). Z3 evaluates 3 < 5
        // directly and returns Proved.
        let term = cmp(CmpOp::Lt, lit(3), lit(5));
        assert_eq!(prove_with_facts(&term, &[], &[]), ProveOutcome::Proved);
    }

    #[test]
    fn cmp_with_matching_input_fact_proves() {
        // Sanity: the existing "Proved via facts" behaviour still works
        // when a fact actually mentions the query's var.
        let facts = vec![cmp(CmpOp::Ge, input("k"), lit(0))];
        let term = cmp(CmpOp::Ge, input("k"), lit(-1));
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Proved);
    }
}
