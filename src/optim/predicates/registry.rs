//! Predicate kind registry — maps a predicate name (`"nnz"`, `"is_sorted"`,
//! …) to its Z3 encoder + universal meta-facts.
//!
//! Per-predicate cards (W1–W5 in the epic) extend this registry by adding
//! entries to [`build_registry`]. The registry is read-only at runtime;
//! initialised once via `OnceLock` on first call to [`registry`].
//!
//! ## What goes in a registration?
//!
//! - `kind` — the string identifier matched against `IR::StructuralPredicate.kind`.
//! - `arity` — number of arguments. The structural-predicate frontend
//!   validates this at decoration time; the encoder double-checks and
//!   degrades gracefully (returns `None`) on arity mismatch.
//! - `encode` — given symbolic `Int` args, returns the predicate's `Bool`
//!   application term. For opaque predicates (no per-element semantics
//!   modelable in pure Z3) this returns a *cached uninterpreted Bool
//!   symbol* keyed by `(kind, args)` — see [`cached_uninterpreted_predicate`].
//! - `meta_facts` — universally-true axioms keyed to the arg(s). Injected
//!   by [`crate::optim::smt_encoding::SmtEncodingCtx::inject_meta_facts`]
//!   with per-kind dedup so repeated references in one query do not
//!   multiply the formula.
//!
//! ## Soundness — what is encoded vs. what is not
//!
//! For each opaque predicate the registry produces an *uninterpreted*
//! Z3 Bool symbol. Same `(kind, args)` always yields the same Bool, so a
//! fact `is_sorted(arr_X)` and a query `is_sorted(arr_X)` line up;
//! different args yield different Bools so `is_sorted(arr_X)` does NOT
//! discharge `is_sorted(arr_Y)`.
//!
//! What is **NOT** encoded:
//! - Per-element semantics (`is_sorted(arr) ⇒ arr[i] <= arr[i+1]`). The
//!   discharger's per-array Int symbol has no array theory; Z3 cannot
//!   reason about elements through it.
//! - Length-relative bounds (`nnz(arr) <= len(arr)`). Would require a
//!   `len_of` ContractTerm operator — filed as
//!   `compiler.contract-term-len-of-operator`.
//! - Quantified properties / array-theory axioms. Out of scope per
//!   project direction.
//!
//! What **IS** encoded:
//! - Standalone numeric bounds on each predicate's value (e.g.,
//!   `nnz >= 0`, `cycle_count >= 1`). Attached to a unique pred-value
//!   Int symbol minted by [`cached_pred_value`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

use z3::ast::{Bool, Int};

/// A single predicate registration.
pub struct PredicateRegistration {
    pub kind: &'static str,
    pub arity: usize,
    encode_fn: fn(&[Int]) -> Bool,
    meta_facts_fn: fn(&[Int]) -> Vec<Bool>,
}

impl PredicateRegistration {
    /// Build the predicate's Z3 application `Bool` over the supplied
    /// symbolic `Int` args.
    pub fn encode_app(&self, args: &[Int]) -> Bool {
        (self.encode_fn)(args)
    }

    /// Universal axioms about this predicate, parameterised by the
    /// invocation's args. Called at most once per (predicate, query) via
    /// `SmtEncodingCtx::inject_meta_facts`.
    pub fn meta_facts(&self, args: &[Int]) -> Vec<Bool> {
        (self.meta_facts_fn)(args)
    }
}

// ---------------------------------------------------------------------------
// cached_uninterpreted_predicate — soundness chokepoint
// ---------------------------------------------------------------------------

thread_local! {
    /// Thread-local cache mapping `(kind, args)` to its uninterpreted Bool
    /// symbol. Bool entries hash by Z3 AST identity; same args (same `Int`
    /// instances coming from a shared `Substitution`) hit the same entry.
    static PREDICATE_BOOL_CACHE: RefCell<HashMap<(&'static str, Vec<Int>), Bool>>
        = RefCell::new(HashMap::new());

    /// Thread-local cache mapping `(kind, args)` to a fresh `Int` symbol
    /// standing for the predicate's *value* (the count, the bound's
    /// other side, etc.). Bounds attached here are sound because the
    /// symbol is dedicated to the predicate, not shared with the array.
    static PREDICATE_VALUE_CACHE: RefCell<HashMap<(&'static str, Vec<Int>), Int>>
        = RefCell::new(HashMap::new());
}

/// Return a cached uninterpreted `Bool` for `(kind, args)`.
///
/// Soundness obligation: this is the chokepoint that replaces the old
/// `v._eq(v)` tautology. With the tautology, `prove(predicate(any_arr))`
/// returned `Proved` for the empty fact set (`facts ∧ ¬True = False`,
/// unsat). With the uninterpreted symbol, `prove()` returns `Proved` only
/// when the matching `predicate(arr_X)` fact is in scope (the lowered
/// Bool symbol then appears in both `facts` and `term`, and Z3 derives
/// the entailment).
///
/// The cache is keyed by `(kind, args)` so same-args calls within one
/// `prove()` query reuse the same `Bool` (correct composition) while
/// different args mint fresh `Bool` symbols (no false unification across
/// unrelated arrays).
fn cached_uninterpreted_predicate(kind: &'static str, args: &[Int]) -> Bool {
    PREDICATE_BOOL_CACHE.with(|cache| {
        let key = (kind, args.to_vec());
        if let Some(b) = cache.borrow().get(&key) {
            return b.clone();
        }
        let fresh = Bool::fresh_const(&format!("pred_{kind}_"));
        cache.borrow_mut().insert(key, fresh.clone());
        fresh
    })
}

/// Return a cached `Int` symbol standing for the predicate's *value*
/// (e.g., the cardinality for `nnz`, the count for `popcount`, etc.).
///
/// Used as the receiver of standalone numeric meta-facts so the bound
/// applies to the predicate's semantic value, not to the array's opaque
/// Int symbol. Same `(kind, args)` yields the same `Int` within a thread.
fn cached_pred_value(kind: &'static str, args: &[Int]) -> Int {
    PREDICATE_VALUE_CACHE.with(|cache| {
        let key = (kind, args.to_vec());
        if let Some(i) = cache.borrow().get(&key) {
            return i.clone();
        }
        let fresh = Int::fresh_const(&format!("pred_value_{kind}_"));
        cache.borrow_mut().insert(key, fresh.clone());
        fresh
    })
}

/// Return the global predicate registry. Lazily initialised on first call.
pub fn registry() -> &'static HashMap<&'static str, PredicateRegistration> {
    static REGISTRY: OnceLock<HashMap<&'static str, PredicateRegistration>> = OnceLock::new();
    REGISTRY.get_or_init(build_registry)
}

/// Build the registry table. Per-predicate cards extend this function.
///
/// ### How to add a new predicate (downstream-card recipe)
///
/// 1. Pick the `kind` string (must match the `zinnia.spec.predicates`
///    Python marker name).
/// 2. Write the `encode_fn` returning a `Bool`. For opaque predicates,
///    delegate to [`cached_uninterpreted_predicate`] — this is the
///    soundness-critical path.
/// 3. Write the `meta_facts_fn` returning `Vec<Bool>`. Each entry is a
///    standalone axiom asserted on first use. Attach bounds to
///    [`cached_pred_value`]`(kind, args)` rather than `args[0]` so the
///    bound applies to the predicate's value, not the array symbol.
/// 4. Insert via `m.insert("kind", PredicateRegistration { … });`.
///
/// Soundness obligation: every `meta_facts` entry must be a tautology
/// under the predicate's intended semantics. Reviewers should re-derive
/// each one from the predicate's definition before approving the PR.
fn build_registry() -> HashMap<&'static str, PredicateRegistration> {
    let mut m: HashMap<&'static str, PredicateRegistration> = HashMap::new();

    // ── nnz ──────────────────────────────────────────────────────────
    //
    // `nnz(arr) op K`: count of non-zero entries. Encoder: uninterpreted
    // Bool keyed by `(nnz, args)`. Meta-fact: `pred_value(nnz, args) >= 0`.
    // Tight upper bound `<= len(arr)` lives in discharge.rs (which has
    // input lengths in scope).
    m.insert("nnz", PredicateRegistration {
        kind: "nnz",
        arity: 1,
        // Soundness: cached uninterpreted Bool, NOT a tautology.
        encode_fn: |args| cached_uninterpreted_predicate("nnz", args),
        meta_facts_fn: |args| {
            let v = cached_pred_value("nnz", args);
            vec![v.ge(&Int::from_i64(0))]
        },
    });

    // ── is_sorted / is_monotone_nondecreasing ─────────────────────────
    //
    // `is_monotone_nondecreasing` is registered as an alias — its encoder
    // routes through the same `"is_sorted"` cache key so both names
    // resolve to the SAME Z3 Bool symbol. A fact about one is a fact
    // about the other.
    //
    // No meta-facts: the predicate doesn't carry a numeric value.
    m.insert("is_sorted", PredicateRegistration {
        kind: "is_sorted",
        arity: 1,
        // Soundness: cached uninterpreted Bool.
        encode_fn: |args| cached_uninterpreted_predicate("is_sorted", args),
        meta_facts_fn: |_args| Vec::new(),
    });
    m.insert("is_monotone_nondecreasing", PredicateRegistration {
        kind: "is_monotone_nondecreasing",
        arity: 1,
        // Soundness: alias to `is_sorted` — same cache key, same Bool.
        encode_fn: |args| cached_uninterpreted_predicate("is_sorted", args),
        meta_facts_fn: |_args| Vec::new(),
    });

    // ── max_run ──────────────────────────────────────────────────────
    //
    // `max_run(arr) op K`: maximum gap between consecutive elements of a
    // monotone array. Meta-fact: `pred_value >= 0` (gaps in a
    // monotone-nondecreasing array are non-negative).
    m.insert("max_run", PredicateRegistration {
        kind: "max_run",
        arity: 1,
        // Soundness: cached uninterpreted Bool.
        encode_fn: |args| cached_uninterpreted_predicate("max_run", args),
        meta_facts_fn: |args| {
            let v = cached_pred_value("max_run", args);
            vec![v.ge(&Int::from_i64(0))]
        },
    });

    // ── is_permutation ───────────────────────────────────────────────
    //
    // `is_permutation(p)` asserts that `p` is a bijection on `[0, N)`.
    // Opaque Bool; per-element semantics live in the witness emitter.
    // No standalone numeric meta-facts.
    m.insert("is_permutation", PredicateRegistration {
        kind: "is_permutation",
        arity: 1,
        // Soundness: cached uninterpreted Bool.
        encode_fn: |args| cached_uninterpreted_predicate("is_permutation", args),
        meta_facts_fn: |_args| Vec::new(),
    });

    // ── cycle_count ──────────────────────────────────────────────────
    //
    // `cycle_count(p) op K`: count of cycles in the permutation graph
    // of `p`. Meta-fact: `pred_value >= 1` (every permutation has at
    // least one cycle). Length-relative upper bound lives in discharge.rs.
    m.insert("cycle_count", PredicateRegistration {
        kind: "cycle_count",
        arity: 1,
        // Soundness: cached uninterpreted Bool.
        encode_fn: |args| cached_uninterpreted_predicate("cycle_count", args),
        meta_facts_fn: |args| {
            let v = cached_pred_value("cycle_count", args);
            vec![v.ge(&Int::from_i64(1))]
        },
    });

    // ── fixed_point_count ────────────────────────────────────────────
    //
    // `fixed_point_count(p) op K`: count of `i` such that `p[i] == i`.
    // Meta-fact: `pred_value >= 0`.
    m.insert("fixed_point_count", PredicateRegistration {
        kind: "fixed_point_count",
        arity: 1,
        // Soundness: cached uninterpreted Bool.
        encode_fn: |args| cached_uninterpreted_predicate("fixed_point_count", args),
        meta_facts_fn: |args| {
            let v = cached_pred_value("fixed_point_count", args);
            vec![v.ge(&Int::from_i64(0))]
        },
    });

    // ── popcount ─────────────────────────────────────────────────────
    //
    // `popcount(b) op K`: cardinality of `true` entries in a boolean
    // array. Meta-fact: `pred_value >= 0`. Tight upper bound `<= len`
    // lives in discharge.rs.
    m.insert("popcount", PredicateRegistration {
        kind: "popcount",
        arity: 1,
        // Soundness: cached uninterpreted Bool.
        encode_fn: |args| cached_uninterpreted_predicate("popcount", args),
        meta_facts_fn: |args| {
            let v = cached_pred_value("popcount", args);
            vec![v.ge(&Int::from_i64(0))]
        },
    });

    // ── is_identity ──────────────────────────────────────────────────
    //
    // `is_identity(arr)` asserts `arr` is a square N×N matrix with
    // `arr[i][j] = 1 iff i == j`. Opaque Bool; per-element semantics live
    // in the producer (`np.identity` ensures). No standalone numeric
    // meta-facts.
    m.insert("is_identity", PredicateRegistration {
        kind: "is_identity",
        arity: 1,
        // Soundness: cached uninterpreted Bool.
        encode_fn: |args| cached_uninterpreted_predicate("is_identity", args),
        meta_facts_fn: |_args| Vec::new(),
    });

    // ── forall_* (synthesized kinds from `forall(arr, P)`) ────────────
    //
    // The user writes `forall(arr, P)` where `P` is one of a closed
    // set of element-predicates (`is_nonneg`, `lt_bound`, `eq_const`,
    // `in_range`). The AST extractor flattens to a synthesized kind
    // (`forall_<inner>`) with the inner predicate's args inlined after
    // the array name.
    //
    // Z3 encoding: cached uninterpreted Bool per (synthesized kind,
    // args). Per-element semantics live in the witness emitter.
    for kind in [
        "forall_is_nonneg",   // arity 1: (arr)
        "forall_lt_bound",    // arity 2: (arr, B)
        "forall_eq_const",    // arity 2: (arr, C)
        "forall_in_range",    // arity 3: (arr, lo, hi)
    ] {
        let encode_fn: fn(&[Int]) -> Bool = match kind {
            "forall_is_nonneg" => |args| cached_uninterpreted_predicate("forall_is_nonneg", args),
            "forall_lt_bound" => |args| cached_uninterpreted_predicate("forall_lt_bound", args),
            "forall_eq_const" => |args| cached_uninterpreted_predicate("forall_eq_const", args),
            "forall_in_range" => |args| cached_uninterpreted_predicate("forall_in_range", args),
            _ => unreachable!(),
        };
        m.insert(kind, PredicateRegistration {
            kind,
            arity: match kind {
                "forall_is_nonneg" => 1,
                "forall_lt_bound" | "forall_eq_const" => 2,
                "forall_in_range" => 3,
                _ => unreachable!(),
            },
            // Soundness: cached uninterpreted Bool per synthesized kind.
            encode_fn,
            meta_facts_fn: |_args| Vec::new(),
        });
    }

    m
}

// ---------------------------------------------------------------------------
// Soundness tests — load-bearing
// ---------------------------------------------------------------------------

#[cfg(test)]
mod soundness_tests {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::predicates::facts::Fact;
    use crate::optim::prove::{prove_with_facts, ProveOutcome};
    use crate::types::ValueId;

    fn vref(p: u64) -> ContractTerm {
        ContractTerm::Var(ContractVar::Value(ValueId(p)))
    }

    fn pred(kind: &str, args: Vec<ContractTerm>) -> ContractTerm {
        ContractTerm::PredicateApp { kind: kind.to_string(), args }
    }

    /// LOAD-BEARING: `prove(predicate(arr_X))` with NO matching fact
    /// must NOT return Proved. This is the regression test that locks
    /// the tautology-encoder soundness bug.
    #[test]
    fn prove_is_sorted_returns_unknown_with_no_fact() {
        let facts: Vec<Fact> = vec![];
        let term = pred("is_sorted", vec![vref(1)]);
        assert_eq!(
            prove_with_facts(&term, &facts, &[]),
            ProveOutcome::Unknown,
            "prove(is_sorted) must be Unknown without a matching fact"
        );
    }

    #[test]
    fn prove_is_sorted_returns_proved_with_matching_fact() {
        let facts = vec![pred("is_sorted", vec![vref(1)])];
        let term = pred("is_sorted", vec![vref(1)]);
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Proved);
    }

    #[test]
    fn prove_is_sorted_returns_unknown_for_different_array() {
        let facts = vec![pred("is_sorted", vec![vref(1)])];
        let term = pred("is_sorted", vec![vref(2)]);
        assert_eq!(
            prove_with_facts(&term, &facts, &[]),
            ProveOutcome::Unknown,
            "different ValueIds must yield different Z3 Bool symbols"
        );
    }

    #[test]
    fn prove_is_sorted_returns_disproved_with_negated_fact() {
        let facts = vec![ContractTerm::Not(Box::new(pred("is_sorted", vec![vref(1)])))];
        let term = pred("is_sorted", vec![vref(1)]);
        assert_eq!(prove_with_facts(&term, &facts, &[]), ProveOutcome::Disproved);
    }

    /// `is_monotone_nondecreasing` aliases to `is_sorted` — a fact about
    /// one must discharge a query about the other (same uninterpreted
    /// Bool symbol).
    #[test]
    fn is_monotone_nondecreasing_aliases_to_is_sorted() {
        let facts = vec![pred("is_sorted", vec![vref(1)])];
        let term = pred("is_monotone_nondecreasing", vec![vref(1)]);
        assert_eq!(
            prove_with_facts(&term, &facts, &[]),
            ProveOutcome::Proved,
            "is_monotone_nondecreasing must alias is_sorted"
        );
    }
}
