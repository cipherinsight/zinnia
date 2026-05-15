//! Discharge orchestrator — the entry point per-card downstream callers
//! invoke when they want to consult structural-predicate facts.
//!
//! ## Responsibilities
//!
//! 1. **Find in-scope structural-predicate atoms** in the IR (these are
//!    global facts; they constrain the program regardless of which
//!    chokepoint is currently being resolved).
//! 2. **Bridge predicate args to Z3 terms** — each `IR::StructuralPredicate`
//!    references input parameters by name; the discharger looks up the
//!    producing `ReadInteger`/`ReadFloat` statement and constructs the
//!    Z3 term for it.
//! 3. **Assemble the formula** — predicate Z3 encodings (uninterpreted
//!    function applications with their symbolic value) plus meta-facts.
//! 4. **Cache results** via [`crate::optim::predicates::cache`].
//!
//! ## What it does NOT do (yet)
//!
//! - Invoke Z3 itself on a complete formula. The existing
//!   [`crate::optim::resolver::SmtResolver`] already owns the Z3 solver
//!   loop and reverse-reachability walker; the contracts card's job is
//!   to surface a parallel set of constraints that the resolver layer
//!   adds to its assembled formula. Wiring the discharger into the
//!   resolver's hot path is the integration step in the next milestone.
//! - Walk the def-use chain to instantiate per-IR-op contracts. The
//!   per-op contract registry is stub-empty in this card; once
//!   per-predicate cards populate it, the discharger gains a
//!   `collect_contracts_along(target, stmts)` pass that walks back from
//!   target and includes each op's contract.
//!
//! ## Modular extension points
//!
//! - `find_structural_predicates(stmts)` is the input collector — works
//!   on any `&[IRStatement]` and returns a flat list. Used by both the
//!   discharger and tests.
//! - `build_input_name_index(stmts)` maps `input_name -> producer StmtId`.
//!   Used to bridge predicate `args` (strings) to existing SSA Z3 terms.
//! - `Discharger::collect_predicate_constraints` is the heart of the
//!   discharge step. It is solver-agnostic up to the `z3` crate types it
//!   returns.

use std::collections::HashMap;

use z3::ast::{Bool, Int};

use crate::circuit_input::PathSegment;
use crate::ir::IRStatement;
use crate::ir_defs::IR;
use crate::optim::predicates::cache::{DischargeCache, DischargeKey, DischargeResult};
use crate::optim::predicates::registry::registry;
use crate::optim::smt_encoding::SmtEncodingCtx;
use crate::types::StmtId;

// ---------------------------------------------------------------------------
// Helpers — pure functions over IR statements
// ---------------------------------------------------------------------------

/// Find every `IR::StructuralPredicate` in `stmts`. Returns `(stmt_id, &IR)`
/// pairs in IR order. Pure helper; testable in isolation.
pub fn find_structural_predicates(
    stmts: &[IRStatement],
) -> Vec<(StmtId, &IR)> {
    stmts
        .iter()
        .filter_map(|s| match &s.ir {
            IR::StructuralPredicate { .. } => Some((s.stmt_id, &s.ir)),
            _ => None,
        })
        .collect()
}

/// Build a map from input-parameter name to the IR statement id that
/// reads it.
///
/// Scalar inputs (`int`, `bool`) produce a single `IR::ReadInteger`
/// statement whose `path.param` is the parameter name and whose
/// `path.segments` is empty. Composite inputs (`NDArray`, `Tuple`, …)
/// produce one read statement per leaf, so the map only contains entries
/// for *scalar* inputs (the foundation contracts card only handles
/// scalar bounds; per-predicate cards extend for array-valued lookups).
pub fn build_input_name_index(stmts: &[IRStatement]) -> HashMap<String, StmtId> {
    let mut idx = HashMap::new();
    for s in stmts {
        if let IR::ReadInteger { path, .. } | IR::ReadFloat { path, .. } = &s.ir {
            if path.segments.is_empty() {
                idx.entry(path.param.clone()).or_insert(s.stmt_id);
            }
        }
    }
    idx
}

/// Derive the static length of each array input by scanning element-wise
/// reads. For an input `x: NDArray[Float, N]`, each element read carries
/// a `path = { param: "x", segments: [Index(i)] }`; the highest `i` plus
/// one is the array's length.
///
/// Returns `name → length` for every input that has at least one
/// `Index(_)`-segmented read. Scalar inputs (no segments) are absent
/// from the map.
///
/// Limitations:
///
/// - Sparse access patterns produce an under-count (the map reflects the
///   max accessed index, not the declared length). For the W1 demo and
///   the standard ir-gen lowering this is fine — the input-parsing
///   prelude reads every leaf. Future work threads the declared shape
///   directly so sparse-read programs are equally well-served.
/// - Only the leading `Index(_)` segment is consulted; multi-D arrays
///   surface as the product of declared dims via the prelude, so the
///   max-`Index` heuristic still recovers the flat total.
pub fn build_input_array_lengths(stmts: &[IRStatement]) -> HashMap<String, u32> {
    let mut max_index: HashMap<String, u32> = HashMap::new();
    for s in stmts {
        let path = match &s.ir {
            IR::ReadInteger { path, .. } | IR::ReadFloat { path, .. } => path,
            _ => continue,
        };
        let first_idx = match path.segments.first() {
            Some(PathSegment::Index(i)) => *i,
            _ => continue,
        };
        let cur = max_index.entry(path.param.clone()).or_insert(0);
        if first_idx >= *cur {
            *cur = first_idx;
        }
    }
    // Convert max-index → length (exclusive bound).
    max_index.into_iter().map(|(k, v)| (k, v + 1)).collect()
}

/// Collect every `IR::ScalarPrecondition` atom from `stmts`. The
/// `term_json` strings are returned as-is; deserialization happens at
/// the discharge call site (via `serde_json::from_str` on the
/// `ContractTerm` type).
pub fn find_scalar_preconditions<'a>(
    stmts: &'a [IRStatement],
) -> Vec<&'a str> {
    stmts
        .iter()
        .filter_map(|s| match &s.ir {
            IR::ScalarPrecondition { term_json } => Some(term_json.as_str()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// PredicateConstraints — the discharger's output payload
// ---------------------------------------------------------------------------

/// What a single `IR::StructuralPredicate` atom contributes to the SMT
/// formula at discharge time.
#[derive(Debug, Clone)]
pub struct PredicateConstraints {
    /// One Bool per predicate atom, encoding its application + any
    /// comparison-with-bound clause. Asserted directly on the solver.
    pub clauses: Vec<Bool>,
    /// Meta-facts contributed by each invoked predicate, deduped by
    /// kind. Caller asserts these in addition to `clauses`.
    pub meta_facts: Vec<Bool>,
}

impl PredicateConstraints {
    fn empty() -> Self {
        Self {
            clauses: Vec::new(),
            meta_facts: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Discharger
// ---------------------------------------------------------------------------

/// The discharge orchestrator. Owns the result cache; stateless otherwise.
#[derive(Debug, Default)]
pub struct Discharger {
    cache: DischargeCache,
}

impl Discharger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Walk all `IR::StructuralPredicate` atoms in `stmts` and return the
    /// Z3 constraints they contribute to a discharge query.
    ///
    /// `arg_z3_for` resolves *scalar* names (input parameters and numeric
    /// literals) to Z3 Int terms. The standard caller passes the walker's
    /// encoding for scalar inputs and `Int::from_i64` for literals.
    ///
    /// Array args are recognised by presence in `input_lengths`. For an
    /// array arg, the discharger:
    ///
    /// 1. Mints a deterministic per-query Z3 symbol for the predicate
    ///    application (e.g., `nnz_x_<unique>`). Repeated references to
    ///    the same `(kind, array_name)` within one query reuse the same
    ///    symbol so Z3 reasons about them as identical.
    /// 2. Injects the length bound (`pred_value >= 0`, `pred_value <=
    ///    len(array)`) — this is the load-bearing fact that lets Z3
    ///    derive bounds on scalars compared against the predicate.
    pub fn collect_predicate_constraints<F>(
        &self,
        stmts: &[IRStatement],
        input_lengths: &HashMap<String, u32>,
        mut arg_z3_for: F,
    ) -> PredicateConstraints
    where
        F: FnMut(&str) -> Int,
    {
        let mut out = PredicateConstraints::empty();
        let mut enc_ctx = SmtEncodingCtx::new();

        // Per-query memo of predicate-application symbols, keyed by
        // (kind, args). Ensures that two references to `nnz(x)` in one
        // query produce the same Z3 Int, so a chain of facts about it
        // composes.
        let mut pred_symbols: HashMap<(String, Vec<String>), Int> = HashMap::new();

        for (_, ir) in find_structural_predicates(stmts) {
            let (kind, args, op, bound) = match ir {
                IR::StructuralPredicate { kind, args, op, bound } => {
                    (kind, args, op, bound)
                }
                _ => unreachable!(),
            };

            let reg = match registry().get(kind.as_str()) {
                Some(r) => r,
                None => continue, // unknown kinds are inert (sound but imprecise)
            };
            if args.len() != reg.arity {
                continue;
            }

            // Build per-arg Z3 Int terms. Array args get a *per-query
            // symbolic* (so the registry's encode_app sees a consistent
            // Int across the meta-facts and the predicate clauses); scalar
            // args / literals go through the caller's closure.
            let int_args: Vec<Int> = args
                .iter()
                .map(|name| {
                    if input_lengths.contains_key(name) {
                        Int::fresh_const(&format!("sp_arr_{name}_"))
                    } else {
                        arg_z3_for(name)
                    }
                })
                .collect();

            // The predicate's application term (a Bool; today a tautology
            // for the nnz stub; per-predicate cards override with real
            // structural content).
            out.clauses.push(reg.encode_app(&int_args));

            // Deterministic per-(kind, args) predicate-value symbol —
            // memoised so chained facts share the same Int.
            let key = (kind.clone(), args.clone());
            let pred_value = pred_symbols
                .entry(key)
                .or_insert_with(|| {
                    Int::fresh_const(&format!(
                        "pred_value_{kind}_{}_",
                        args.join("_")
                    ))
                })
                .clone();

            // Predicate-value semantic bounds. Each kind whose value is
            // an Int derives a known range from the predicate's
            // definition. The discharger owns this mapping (registry's
            // `meta_facts_fn` operates on array symbols, not on the
            // predicate-value symbol, so the bounds live here).
            match kind.as_str() {
                // Cardinality predicates: `0 <= pred <= len(arr)`.
                "nnz" | "popcount" => {
                    if let Some(&len) = args.first().and_then(|n| input_lengths.get(n)) {
                        out.clauses.push(pred_value.ge(&Int::from_i64(0)));
                        out.clauses.push(pred_value.le(&Int::from_i64(len as i64)));
                    }
                }
                // Cycle / fixed-point counts on permutations: `1 <= cycle_count <= len`,
                // `0 <= fixed_point_count <= len`. Registered for completeness
                // ahead of `compiler.structural-predicate-permutation`.
                "cycle_count" => {
                    if let Some(&len) = args.first().and_then(|n| input_lengths.get(n)) {
                        if len > 0 {
                            out.clauses.push(pred_value.ge(&Int::from_i64(1)));
                            out.clauses.push(pred_value.le(&Int::from_i64(len as i64)));
                        }
                    }
                }
                "fixed_point_count" => {
                    if let Some(&len) = args.first().and_then(|n| input_lengths.get(n)) {
                        out.clauses.push(pred_value.ge(&Int::from_i64(0)));
                        out.clauses.push(pred_value.le(&Int::from_i64(len as i64)));
                    }
                }
                // Max-run on a monotone array: non-negative (gaps in a
                // monotone-nondecreasing sequence are >= 0). No tight
                // upper bound from length alone — the user supplies the
                // bound via `max_run(arr) <= K`.
                "max_run" => {
                    out.clauses.push(pred_value.ge(&Int::from_i64(0)));
                }
                _ => {}
            }

            // Comparison-with-bound clause: tie the predicate's value to
            // the bound. The bound may be a scalar input (resolved via
            // `arg_z3_for`) or a numeric literal.
            if let (Some(op_str), Some(bound_name)) = (op.as_deref(), bound.as_deref()) {
                let bound_int = arg_z3_for(bound_name);
                if let Some(clause) = compare(&pred_value, op_str, &bound_int) {
                    out.clauses.push(clause);
                }
            }

            // Meta-facts for this predicate (deduped via SmtEncodingCtx).
            let facts = reg.meta_facts(&int_args);
            enc_ctx.inject_meta_facts(kind.as_str(), facts);
        }

        out.meta_facts = std::mem::take(&mut enc_ctx.meta_facts);
        out
    }

    /// Cache accessor — exposed for tests and future integration points.
    pub fn cache(&self) -> &DischargeCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut DischargeCache {
        &mut self.cache
    }

    /// Memoised discharge entry. Cards that wire this into a real solver
    /// pipeline call `try_discharge_with(...)` after running the actual
    /// Z3 query; the cache stores the outcome under a content-derived
    /// key so repeated identical queries are O(1).
    pub fn try_discharge_with<F: FnOnce() -> DischargeResult>(
        &mut self,
        key: DischargeKey,
        run: F,
    ) -> DischargeResult {
        if let Some(cached) = self.cache.get(&key) {
            return cached;
        }
        let result = run();
        self.cache.insert(key, result);
        result
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compare(lhs: &Int, op: &str, rhs: &Int) -> Option<Bool> {
    Some(match op {
        "==" => lhs.eq(rhs),
        "!=" => lhs.eq(rhs).not(),
        "<" => lhs.lt(rhs),
        "<=" => lhs.le(rhs),
        ">" => lhs.gt(rhs),
        ">=" => lhs.ge(rhs),
        _ => return None,
    })
}
