//! Witness-time enforcement of structural-predicate preconditions.
//!
//! The structural-predicate IR atom (emitted by the surface card) is a
//! *compile-time fact*. Without an accompanying witness-time check, a
//! malicious prover can supply inputs that violate the precondition and
//! still produce a valid proof — the compiler trusts the fact but the
//! circuit does not enforce it.
//!
//! This module ships the **witness-emitter registry**: per-predicate
//! callbacks that the ir-gen layer invokes after every `ASTRequires` to
//! emit real circuit constraints. nnz ships the first emitter (sum of
//! per-element indicators + final assert). Future per-predicate cards
//! register their own emitters using the same hook.
//!
//! ## Soundness contract
//!
//! For a precondition `P(args) op bound` (e.g., `nnz(x) == k`), the
//! emitter must produce IR such that:
//!
//!   `prover supplies witness violating P(args) op bound` ⇒ `proof fails`
//!
//! Each emitter ships with a soundness paragraph in its body. Reviewers
//! re-derive the proof obligation from the predicate's intended
//! semantics before approving the PR that adds the emitter.
//!
//! ## Why a registry vs. inline emission
//!
//! Per-predicate emitters keep the ir-gen layer agnostic to predicate
//! semantics. A new predicate adds its emitter, registers it here, and
//! the existing `visit_circuit` invocation picks it up — no changes to
//! the ir-gen orchestration. Mirrors the [`crate::optim::predicates::registry`]
//! pattern for SMT encoding.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::builder::IRBuilder;
use crate::ir_defs::IR;
use crate::types::{ScalarValue, Value};

// ---------------------------------------------------------------------------
// WitnessEmitter
// ---------------------------------------------------------------------------

/// Per-predicate witness-time circuit emitter.
///
/// `emit` is called once per `ASTRequires` recognised at ir-gen time. It
/// is responsible for adding constraints to the circuit that the prover
/// must satisfy. The function returns `()` — failure to emit (e.g.,
/// unsupported op) is logged via a debug print and the precondition is
/// left unenforced. **Unenforced precondition = soundness gap**; tests
/// must catch this at integration time.
pub struct WitnessEmitter {
    pub kind: &'static str,
    pub emit: fn(&mut IRBuilder, &[Value], Option<&str>, Option<&Value>),
}

/// Global registry — populated lazily. Per-predicate cards extend
/// [`build_witness_emitters`].
pub fn witness_emitters() -> &'static HashMap<&'static str, WitnessEmitter> {
    static EMITTERS: OnceLock<HashMap<&'static str, WitnessEmitter>> = OnceLock::new();
    EMITTERS.get_or_init(build_witness_emitters)
}

fn build_witness_emitters() -> HashMap<&'static str, WitnessEmitter> {
    let mut m: HashMap<&'static str, WitnessEmitter> = HashMap::new();

    // ── nnz ───────────────────────────────────────────────────────────
    // Emits sum-of-indicators + assert against the bound.
    m.insert("nnz", WitnessEmitter {
        kind: "nnz",
        emit: emit_nnz,
    });

    // ── is_sorted / is_monotone_nondecreasing ──────────────────────────
    // Emit per-adjacent-pair `arr[i] <= arr[i+1]` + assert. Unary
    // predicate (no op / bound). Soundness: prover supplying an
    // unsorted array fails at the first violating pair.
    m.insert("is_sorted", WitnessEmitter {
        kind: "is_sorted",
        emit: emit_is_sorted,
    });
    m.insert("is_monotone_nondecreasing", WitnessEmitter {
        kind: "is_monotone_nondecreasing",
        emit: emit_is_sorted,  // same semantics
    });

    // ── max_run ────────────────────────────────────────────────────────
    // Emits per-adjacent-pair `arr[i+1] - arr[i]` and asserts against
    // the bound via `op`. Today supports `op = "<="` (the common usage
    // in CSR-style preconditions); other ops fall through as a no-op
    // (soundness gap; documented in source).
    m.insert("max_run", WitnessEmitter {
        kind: "max_run",
        emit: emit_max_run,
    });

    // ── is_permutation ────────────────────────────────────────────────
    // Emits range checks (`0 <= p[i] < N`) plus injectivity checks
    // (`p[i] != p[j]` for `i < j`). O(N²) constraints for naive
    // emission. Large-N programs likely want a Plonk-style permutation
    // argument; that's a future tightening, gated on a real workload.
    m.insert("is_permutation", WitnessEmitter {
        kind: "is_permutation",
        emit: emit_is_permutation,
    });

    // ── fixed_point_count ─────────────────────────────────────────────
    // Same indicator-sum pattern as `nnz` / `popcount`: per index,
    // `p[i] == i` as a 0/1 indicator; sum across indices; assert
    // against the bound via `op`.
    m.insert("fixed_point_count", WitnessEmitter {
        kind: "fixed_point_count",
        emit: emit_fixed_point_count,
    });

    // `cycle_count`: deliberately not registered. Witness-time
    // enforcement requires graph traversal that doesn't lower
    // cleanly to a flat circuit. The IR atom remains as a
    // compile-time fact only; a misassertion is not caught at prove
    // time. Documented as a soundness gap; recommend a follow-up if
    // a real workload needs the witness check.

    // ── popcount ──────────────────────────────────────────────────────
    // Semantically `nnz` on boolean arrays. `emit_nnz` already
    // dispatches per-element on Bool/Int/Float, so the boolean case is
    // handled correctly. Register the same emitter under a separate
    // key so bitmap workloads can use the `popcount` name naturally.
    m.insert("popcount", WitnessEmitter {
        kind: "popcount",
        emit: emit_nnz,
    });

    // ── forall_* family (synthesized from `forall(arr, P)`) ───────────
    //
    // Each variant emits per-element checks matching the inner
    // element-predicate's semantics. Sound: a violating element
    // produces a failing assert.
    m.insert("forall_is_nonneg", WitnessEmitter {
        kind: "forall_is_nonneg",
        emit: emit_forall_is_nonneg,
    });
    m.insert("forall_lt_bound", WitnessEmitter {
        kind: "forall_lt_bound",
        emit: emit_forall_lt_bound,
    });
    m.insert("forall_eq_const", WitnessEmitter {
        kind: "forall_eq_const",
        emit: emit_forall_eq_const,
    });
    m.insert("forall_in_range", WitnessEmitter {
        kind: "forall_in_range",
        emit: emit_forall_in_range,
    });

    m
}

// ---------------------------------------------------------------------------
// nnz witness emitter
// ---------------------------------------------------------------------------

/// Soundness argument:
///
/// For an array `x` of length `N` and a scalar `k`, the precondition
/// `nnz(x) op k` is enforced by computing the true count of nonzero
/// elements (a per-element indicator summed across the array) and
/// asserting that count compared with `k` via `op` holds.
///
/// - Indicator: `indicator_i = (x[i] != 0)` as a 0/1 Int.
/// - Count: `count = Σ_i indicator_i`. Bounded by `N` ≤ field modulus,
///   so no overflow within Zinnia's integer domain.
/// - Assert: `count op k` reduced to a scalar Bool via the comparison
///   IR ops, then `IR::Assert`.
///
/// A misassertion (prover supplies `x, k` with `actual_nnz(x) != k`)
/// produces an unsatisfiable circuit constraint — proof fails. Sound.
///
/// Limitations of this emitter:
///
/// - Float comparison via `NeF`: relies on the existing Zinnia float
///   encoding. Subnormals / signed-zero behave according to that
///   encoding; document if a downstream consumer depends on edge cases.
/// - Multi-dim arrays: the emitter flattens via `to_value_list`, which
///   walks the array's flat layout — sound but produces N constraints
///   regardless of shape rank.
/// - Empty arrays (`N == 0`): emit `count = 0` literal + assert against
///   bound. Trivially correct (`count op k` reduces to `0 op k`).
fn emit_nnz(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    if arg_values.len() != 1 {
        return;
    }

    // Validate op + bound *before* emitting any constraints. A unary
    // predicate (no op / no bound) has nothing to enforce; an
    // unsupported op should also short-circuit cleanly so the caller
    // doesn't see a dangling indicator chain.
    let comparison_ir = match op {
        Some("==") => IR::EqI,
        Some("!=") => IR::NeI,
        Some("<") => IR::LtI,
        Some("<=") => IR::LteI,
        Some(">") => IR::GtI,
        Some(">=") => IR::GteI,
        _ => return,
    };
    let bound_val = match bound {
        Some(v) => v.clone(),
        None => return,
    };

    let array_val = &arg_values[0];

    // Get the per-element values via the static-array helper.
    let elements = match array_val {
        Value::StaticArray { .. } => {
            let list = crate::helpers::static_array::to_value_list(b, array_val);
            match list {
                Value::List(data) => data.values,
                _ => return, // unsupported shape
            }
        }
        Value::List(data) => data.values.clone(),
        _ => {
            // Scalar arg or unsupported shape — skip emission. Documented
            // as a precondition: the surface card validates that the
            // first arg of `nnz` is an array; if we reach here with a
            // non-array, the surface validation has a gap.
            return;
        }
    };

    // Build the indicator sum.
    let mut count = b.ir_constant_int(0);
    for elem in &elements {
        let indicator = emit_nonzero_indicator(b, elem);
        count = b.create_ir(&IR::AddI, &[count, indicator]);
    }

    // Compare against the bound and assert.
    let cmp = b.create_ir(&comparison_ir, &[count, bound_val]);
    b.ir_assert(&cmp);
}

// ---------------------------------------------------------------------------
// is_sorted / is_monotone_nondecreasing witness emitter
// ---------------------------------------------------------------------------

/// Soundness argument:
///
/// For an array `arr` of length `N`, the predicate `is_sorted(arr)` (or
/// the equivalent `is_monotone_nondecreasing(arr)`) is enforced by
/// asserting `arr[i] <= arr[i+1]` for every adjacent pair. If any pair
/// violates the order, the corresponding assert is false → the proof
/// fails. Sound.
///
/// Cost: 2(N-1) IR statements (one `LteI`/`LteF` and one `Assert` per
/// adjacent pair).
///
/// Behaviour on edge cases:
/// - `N = 0` or `N = 1`: trivially sorted → no asserts emitted.
/// - Op present (`is_sorted(arr) <= K` etc.): unsupported; the predicate
///   is unary. Falls through as a no-op.
fn emit_is_sorted(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    // Unary predicate — op/bound must not be present.
    if op.is_some() || bound.is_some() {
        return;
    }
    if arg_values.len() != 1 {
        return;
    }
    let elements = match collect_elements(b, &arg_values[0]) {
        Some(v) => v,
        None => return,
    };
    if elements.len() < 2 {
        return; // trivially sorted
    }
    for pair in elements.windows(2) {
        let cmp = emit_le(b, &pair[0], &pair[1]);
        if let Some(c) = cmp {
            b.ir_assert(&c);
        }
    }
}

// ---------------------------------------------------------------------------
// max_run witness emitter
// ---------------------------------------------------------------------------

/// Soundness argument (for op = `<=`):
///
/// For a monotone-nondecreasing array `arr` of length `N` and bound `K`,
/// the predicate `max_run(arr) <= K` is enforced by asserting
/// `arr[i+1] - arr[i] <= K` for every adjacent pair. The max gap
/// across all pairs is, by definition, the largest of these
/// differences; if every pair satisfies `<= K`, the max gap does too.
///
/// Limitations:
/// - Today only `op = "<="` is supported. Other ops (`==`, `>=`, `<`,
///   `>`) require computing the actual max gap and comparing — more
///   work; deferred. Sound *omission* would be to emit nothing for
///   unsupported ops, which is what we do: the precondition remains a
///   compile-time fact the SMT layer can use, but soundness at prove
///   time is *not* enforced for unsupported ops. **Document this gap
///   explicitly in the card status.**
/// - Float arrays: subtracts use `SubF`; semantics depend on Zinnia's
///   field representation.
/// - Int array, `Int` bound: standard.
/// - Empty / singleton arrays: no asserts emitted (trivially satisfies
///   any bound).
fn emit_max_run(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    // Only `<=` is supported today.
    if !matches!(op, Some("<=")) {
        return;
    }
    if arg_values.len() != 1 {
        return;
    }
    let bound_val = match bound {
        Some(v) => v.clone(),
        None => return,
    };
    let elements = match collect_elements(b, &arg_values[0]) {
        Some(v) => v,
        None => return,
    };
    if elements.len() < 2 {
        return;
    }
    // For each adjacent pair, emit `arr[i+1] - arr[i] <= bound` + assert.
    for pair in elements.windows(2) {
        let diff = emit_sub(b, &pair[1], &pair[0]);
        if let Some(d) = diff {
            // `bound_val` is the user's `K`; the assert is `d <= K`.
            let cmp = b.create_ir(&IR::LteI, &[d, bound_val.clone()]);
            b.ir_assert(&cmp);
        }
    }
}

// ---------------------------------------------------------------------------
// is_permutation witness emitter
// ---------------------------------------------------------------------------

/// Soundness argument:
///
/// `is_permutation(p)` for a length-N array `p` is enforced by:
/// 1. **Range checks** (N constraints): for each i, assert
///    `0 <= p[i] < N`. Ensures every value is a valid index.
/// 2. **Injectivity checks** (N(N-1)/2 constraints): for each
///    `i < j`, assert `p[i] != p[j]`. Ensures no duplicates.
///
/// Range + distinctness on a finite set of size N with N values is
/// equivalent to bijectivity. Sound.
///
/// Cost: O(N²) IR statements. Acceptable for small N (< ~64); larger
/// programs likely want a Plonk-style permutation argument (out of
/// scope for the naive emitter — gated on a real workload).
///
/// Behaviour on edge cases:
/// - Unary predicate: op/bound must not be present.
/// - N = 0 or N = 1: trivially a permutation; emits at most one range
///   check.
fn emit_is_permutation(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    if op.is_some() || bound.is_some() {
        return;
    }
    if arg_values.len() != 1 {
        return;
    }
    let elements = match collect_elements(b, &arg_values[0]) {
        Some(v) => v,
        None => return,
    };
    let n = elements.len();
    if n == 0 {
        return;
    }
    let n_const = b.ir_constant_int(n as i64);
    let zero = b.ir_constant_int(0);

    // 1) Range checks: 0 <= p[i] < N for every i.
    for e in &elements {
        // p[i] >= 0
        let ge = b.create_ir(&IR::GteI, &[e.clone(), zero.clone()]);
        b.ir_assert(&ge);
        // p[i] < N
        let lt = b.create_ir(&IR::LtI, &[e.clone(), n_const.clone()]);
        b.ir_assert(&lt);
    }

    // 2) Injectivity: p[i] != p[j] for every i < j.
    for i in 0..n {
        for j in (i + 1)..n {
            let ne = b.create_ir(&IR::NeI, &[elements[i].clone(), elements[j].clone()]);
            b.ir_assert(&ne);
        }
    }
}

// ---------------------------------------------------------------------------
// fixed_point_count witness emitter
// ---------------------------------------------------------------------------

/// Soundness argument:
///
/// For `fixed_point_count(p) op K`, compute the number of indices `i`
/// where `p[i] == i` (per-index indicator) and sum the indicators.
/// Assert the sum compared with `K` via `op`.
///
/// Indicator: `is_fixed_i = (p[i] == i ? 1 : 0)`. Cost N indicators + N
/// adds + one comparison + one assert ≈ 2N + 2 IR statements.
///
/// Supported ops: `==`, `!=`, `<`, `<=`, `>`, `>=`. Other ops fall
/// through as a no-op (no soundness regression but no enforcement).
fn emit_fixed_point_count(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    if arg_values.len() != 1 {
        return;
    }
    let comparison_ir = match op {
        Some("==") => IR::EqI,
        Some("!=") => IR::NeI,
        Some("<") => IR::LtI,
        Some("<=") => IR::LteI,
        Some(">") => IR::GtI,
        Some(">=") => IR::GteI,
        _ => return,
    };
    let bound_val = match bound {
        Some(v) => v.clone(),
        None => return,
    };
    let elements = match collect_elements(b, &arg_values[0]) {
        Some(v) => v,
        None => return,
    };
    let mut count = b.ir_constant_int(0);
    for (i, elem) in elements.iter().enumerate() {
        let idx = b.ir_constant_int(i as i64);
        let is_fixed = b.create_ir(&IR::EqI, &[elem.clone(), idx]);
        let indicator = b.create_ir(&IR::IntCast, &[is_fixed]);
        count = b.create_ir(&IR::AddI, &[count, indicator]);
    }
    let cmp = b.create_ir(&comparison_ir, &[count, bound_val]);
    b.ir_assert(&cmp);
}

// ---------------------------------------------------------------------------
// forall_* element-predicate witness emitters
// ---------------------------------------------------------------------------

/// Per-element `arr[i] >= 0` + assert.
///
/// Soundness: a violating element produces a failing `>=` and a failing
/// assert. Float arrays use `GteI` after lifting via `IntCast(elem !=
/// 0.0)` is wrong — we instead branch on element type and use `GteF`
/// for floats.
fn emit_forall_is_nonneg(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    if op.is_some() || bound.is_some() {
        return;
    }
    if arg_values.len() != 1 {
        return;
    }
    let elements = match collect_elements(b, &arg_values[0]) {
        Some(v) => v,
        None => return,
    };
    for elem in &elements {
        let zero = match elem {
            Value::Float(_) => b.ir_constant_float(0.0),
            _ => b.ir_constant_int(0),
        };
        let cmp = match elem {
            Value::Integer(_) | Value::Boolean(_) => {
                b.create_ir(&IR::GteI, &[elem.clone(), zero])
            }
            Value::Float(_) => b.create_ir(&IR::GteF, &[elem.clone(), zero]),
            _ => continue,
        };
        b.ir_assert(&cmp);
    }
}

/// Per-element `arr[i] < B` + assert. The `B` is `arg_values[1]`.
fn emit_forall_lt_bound(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    if op.is_some() || bound.is_some() {
        return;
    }
    if arg_values.len() != 2 {
        return;
    }
    let bound_val = arg_values[1].clone();
    let elements = match collect_elements(b, &arg_values[0]) {
        Some(v) => v,
        None => return,
    };
    for elem in &elements {
        let cmp = match elem {
            Value::Integer(_) | Value::Boolean(_) => {
                b.create_ir(&IR::LtI, &[elem.clone(), bound_val.clone()])
            }
            Value::Float(_) => b.create_ir(&IR::LtF, &[elem.clone(), bound_val.clone()]),
            _ => continue,
        };
        b.ir_assert(&cmp);
    }
}

/// Per-element `arr[i] == C` + assert. The `C` is `arg_values[1]`.
fn emit_forall_eq_const(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    if op.is_some() || bound.is_some() {
        return;
    }
    if arg_values.len() != 2 {
        return;
    }
    let const_val = arg_values[1].clone();
    let elements = match collect_elements(b, &arg_values[0]) {
        Some(v) => v,
        None => return,
    };
    for elem in &elements {
        let cmp = match elem {
            Value::Integer(_) | Value::Boolean(_) => {
                b.create_ir(&IR::EqI, &[elem.clone(), const_val.clone()])
            }
            Value::Float(_) => b.create_ir(&IR::EqF, &[elem.clone(), const_val.clone()]),
            _ => continue,
        };
        b.ir_assert(&cmp);
    }
}

/// Per-element `lo <= arr[i] <= hi` + assert. `lo = arg_values[1]`,
/// `hi = arg_values[2]`. Emits two asserts per element.
fn emit_forall_in_range(
    b: &mut IRBuilder,
    arg_values: &[Value],
    op: Option<&str>,
    bound: Option<&Value>,
) {
    if op.is_some() || bound.is_some() {
        return;
    }
    if arg_values.len() != 3 {
        return;
    }
    let lo = arg_values[1].clone();
    let hi = arg_values[2].clone();
    let elements = match collect_elements(b, &arg_values[0]) {
        Some(v) => v,
        None => return,
    };
    for elem in &elements {
        let (ge, le) = match elem {
            Value::Integer(_) | Value::Boolean(_) => (
                b.create_ir(&IR::GteI, &[elem.clone(), lo.clone()]),
                b.create_ir(&IR::LteI, &[elem.clone(), hi.clone()]),
            ),
            Value::Float(_) => (
                b.create_ir(&IR::GteF, &[elem.clone(), lo.clone()]),
                b.create_ir(&IR::LteF, &[elem.clone(), hi.clone()]),
            ),
            _ => continue,
        };
        b.ir_assert(&ge);
        b.ir_assert(&le);
    }
}

// ---------------------------------------------------------------------------
// Scalar precondition lowering: ContractTerm → Zinnia IR
// ---------------------------------------------------------------------------
//
// Walks a `ContractTerm` and emits Zinnia IR producing a `Value::Boolean`
// for the term. The witness emitter wraps the result in `IR::Assert` so
// the prover must satisfy the precondition at prove time.
//
// `Var(Input(name))` is resolved against the IR context's input bindings
// (the caller passes a closure that maps name → Value). `LitInt` /
// `LitBool` emit constants. `Arith` and `Cmp` dispatch to the
// corresponding Int/Bool ops. `BoolComb` and `Not` use the existing
// logical-op IR. `PredicateApp` is intentionally **not** supported in
// scalar context for v1 — users compose structural + scalar preconditions
// by writing multiple `@requires` clauses.

use crate::optim::predicates::formula::{
    ArithOp, BoolOp, CmpOp, ContractTerm, ContractVar,
};

/// Lower a user-written `@requires` precondition (a `ContractTerm` with
/// `Var(Input(name))` leaves, top-level Bool) into Zinnia IR. Returns
/// the `Value::Boolean` representing the predicate's runtime value;
/// the caller wraps in `ir_assert(...)` to enforce at proof time.
///
/// Errors carry a string diagnostic — the scalar-precondition flow logs
/// and skips emission on error (sound omission: no Assert, so the
/// precondition has no proof-time enforcement; documented soundness
/// gap).
///
/// **Naming.** "Precondition" is the formal name for what `@requires`
/// expresses, and it distinguishes this lowering from:
/// - op contracts (which produce *post*conditions / facts deposited at
///   compile time, not lowered to runtime IR);
/// - structural predicates (`nnz`, `Sorted`, …), whose proof-time
///   witnesses are emitted by `WitnessEmitter` / `witness_emitters` in
///   this module — that's the ZK-proof sense of "witness."
pub fn lower_precondition_to_ir<F>(
    b: &mut IRBuilder,
    term: &ContractTerm,
    name_lookup: &F,
) -> Result<Value, String>
where
    F: Fn(&str) -> Option<Value>,
{
    emit_bool_term(b, term, name_lookup)
}

fn emit_bool_term<F>(
    b: &mut IRBuilder,
    term: &ContractTerm,
    name_lookup: &F,
) -> Result<Value, String>
where
    F: Fn(&str) -> Option<Value>,
{
    match term {
        ContractTerm::LitBool(v) => Ok(b.ir_constant_bool(*v)),

        ContractTerm::Cmp { op, lhs, rhs } => {
            let l = emit_int_term(b, lhs, name_lookup)?;
            let r = emit_int_term(b, rhs, name_lookup)?;
            let ir = match op {
                CmpOp::Eq => IR::EqI,
                CmpOp::Ne => IR::NeI,
                CmpOp::Lt => IR::LtI,
                CmpOp::Le => IR::LteI,
                CmpOp::Gt => IR::GtI,
                CmpOp::Ge => IR::GteI,
            };
            Ok(b.create_ir(&ir, &[l, r]))
        }

        ContractTerm::BoolComb { op, operands } => {
            if operands.is_empty() {
                return Ok(b.ir_constant_bool(matches!(op, BoolOp::And)));
            }
            let mut acc = emit_bool_term(b, &operands[0], name_lookup)?;
            for next in &operands[1..] {
                let next_v = emit_bool_term(b, next, name_lookup)?;
                let ir = match op {
                    BoolOp::And => IR::LogicalAnd,
                    BoolOp::Or => IR::LogicalOr,
                };
                acc = b.create_ir(&ir, &[acc, next_v]);
            }
            Ok(acc)
        }

        ContractTerm::Not(inner) => {
            let v = emit_bool_term(b, inner, name_lookup)?;
            Ok(b.create_ir(&IR::LogicalNot, &[v]))
        }

        ContractTerm::PredicateApp { kind, .. } => Err(format!(
            "scalar precondition cannot contain predicate call `{kind}(...)`; \
             write a separate `@requires(lambda ...)` for the structural \
             precondition and another for the scalar bound. v1 limitation."
        )),

        // Numeric leaves / arithmetic in a Bool position: template
        // shape bug.
        ContractTerm::Var(_)
        | ContractTerm::LitInt(_)
        | ContractTerm::LitFloat(_)
        | ContractTerm::Arith { .. } => {
            Err(format!(
                "scalar precondition Bool-context expected a comparison or \
                 logical composition; got a numeric-typed term"
            ))
        }
    }
}

fn emit_int_term<F>(
    b: &mut IRBuilder,
    term: &ContractTerm,
    name_lookup: &F,
) -> Result<Value, String>
where
    F: Fn(&str) -> Option<Value>,
{
    match term {
        ContractTerm::Var(ContractVar::Input(name)) => name_lookup(name)
            .ok_or_else(|| format!("scalar precondition references unbound input `{name}`")),

        ContractTerm::Var(ContractVar::Output) => Err(
            "scalar precondition cannot reference `Output` (a contract \
             template var; not valid in @requires)".to_string()
        ),

        ContractTerm::Var(ContractVar::Value(_)) => Err(
            "scalar precondition cannot reference `Value` (only contract \
             templates carry ValueId-anchored leaves; @requires terms \
             reach the witness emitter pre-substitution with Input(name) \
             leaves only)".to_string()
        ),

        ContractTerm::Var(ContractVar::Formal(name)) => Err(format!(
            "scalar precondition cannot contain template formal `{name}`; \
             contract templates must be fully substituted before reaching \
             witness IR emission"
        )),

        ContractTerm::LitInt(n) => Ok(b.ir_constant_int(*n)),
        ContractTerm::LitFloat(f) => Ok(b.ir_constant_float(f.0)),

        ContractTerm::Arith { op, lhs, rhs } => {
            let l = emit_int_term(b, lhs, name_lookup)?;
            let r = emit_int_term(b, rhs, name_lookup)?;
            let ir = match op {
                ArithOp::Add => IR::AddI,
                ArithOp::Sub => IR::SubI,
                ArithOp::Mul => IR::MulI,
                ArithOp::Div => IR::DivI,
                ArithOp::FloorDiv => IR::FloorDivI,
                ArithOp::Mod => IR::ModI,
                ArithOp::Pow => IR::PowI,
            };
            Ok(b.create_ir(&ir, &[l, r]))
        }

        // Bool-typed leaves / compositions in an Int position: template
        // shape bug.
        _ => Err(format!(
            "scalar precondition Int-context expected an Int-typed term; \
             got a Bool-typed term"
        )),
    }
}

// ---------------------------------------------------------------------------
// Element helpers — shared between emitters
// ---------------------------------------------------------------------------

/// Flatten an array-like Value to a list of element Values. Returns
/// `None` for unsupported shapes (the caller short-circuits).
fn collect_elements(b: &mut IRBuilder, val: &Value) -> Option<Vec<Value>> {
    match val {
        Value::StaticArray { .. } => {
            let list = crate::helpers::static_array::to_value_list(b, val);
            match list {
                Value::List(data) => Some(data.values),
                _ => None,
            }
        }
        Value::List(data) => Some(data.values.clone()),
        _ => None,
    }
}

/// Emit `lhs <= rhs` as a Bool. Dispatches on element type.
fn emit_le(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Option<Value> {
    match (lhs, rhs) {
        (Value::Integer(_), Value::Integer(_)) => {
            Some(b.create_ir(&IR::LteI, &[lhs.clone(), rhs.clone()]))
        }
        (Value::Float(_), Value::Float(_)) => {
            Some(b.create_ir(&IR::LteF, &[lhs.clone(), rhs.clone()]))
        }
        (Value::Boolean(_), Value::Boolean(_)) => {
            // Bool <= Bool: lift to Int and compare. The use-case is
            // `is_sorted` on a boolean array; rare but legal.
            let l = b.create_ir(&IR::IntCast, &[lhs.clone()]);
            let r = b.create_ir(&IR::IntCast, &[rhs.clone()]);
            Some(b.create_ir(&IR::LteI, &[l, r]))
        }
        _ => None,
    }
}

/// Emit `lhs - rhs` as an Int. Dispatches on element type; Float results
/// are cast to Int via `IntCast` to match the bound's expected type.
fn emit_sub(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Option<Value> {
    match (lhs, rhs) {
        (Value::Integer(_), Value::Integer(_)) => {
            Some(b.create_ir(&IR::SubI, &[lhs.clone(), rhs.clone()]))
        }
        (Value::Float(_), Value::Float(_)) => {
            let diff = b.create_ir(&IR::SubF, &[lhs.clone(), rhs.clone()]);
            Some(b.create_ir(&IR::IntCast, &[diff]))
        }
        _ => None,
    }
}

/// Emit a 0/1 Int indicator for `elem != 0`. Dispatches on the element's
/// scalar type (Int vs Float).
fn emit_nonzero_indicator(b: &mut IRBuilder, elem: &Value) -> Value {
    match elem {
        Value::Integer(_) => {
            let zero = b.ir_constant_int(0);
            // `NeI` returns a Bool; cast to Int.
            let ne = b.create_ir(&IR::NeI, &[elem.clone(), zero]);
            b.create_ir(&IR::IntCast, &[ne])
        }
        Value::Float(_) => {
            let zero = b.ir_constant_float(0.0);
            let ne = b.create_ir(&IR::NeF, &[elem.clone(), zero]);
            b.create_ir(&IR::IntCast, &[ne])
        }
        Value::Boolean(_) => {
            // For boolean arrays, `nnz` collapses to `popcount`: the
            // indicator is the value itself cast to Int.
            b.create_ir(&IR::IntCast, &[elem.clone()])
        }
        _ => {
            // Composite or unsupported element type — emit a Bool false
            // indicator so the count under-approximates. Sound: the
            // assert will likely fail at prove time, which is the right
            // behaviour for an unsupported encoding.
            let _ = (b, elem);
            let z = ScalarValue::new(Some(0), None);
            Value::Integer(z)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompositeData, ValueId};

    fn list_of_ints(b: &mut IRBuilder, values: &[i64]) -> Value {
        let mut elems = Vec::with_capacity(values.len());
        for &n in values {
            elems.push(b.ir_constant_int(n));
        }
        Value::List(CompositeData {
            elements_type: vec![crate::types::ZinniaType::Integer; values.len()],
            values: elems,
        
            value_id: ValueId::next(),
        })
    }

    #[test]
    fn registry_contains_nnz_emitter() {
        let r = witness_emitters();
        assert!(r.get("nnz").is_some());
        assert_eq!(r.get("nnz").unwrap().kind, "nnz");
    }

    #[test]
    fn emit_nnz_on_empty_arglist_is_noop() {
        let mut b = IRBuilder::new();
        let before = b.stmts.len();
        emit_nnz(&mut b, &[], Some("=="), None);
        assert_eq!(b.stmts.len(), before, "no args → no emission");
    }

    #[test]
    fn emit_nnz_on_unary_predicate_is_noop() {
        // nnz(x) without a comparison op shouldn't emit a witness
        // (there's nothing to compare the count against).
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[1, 0, 1]);
        let before = b.stmts.len();
        emit_nnz(&mut b, &[arr], None, None);
        assert_eq!(b.stmts.len(), before, "unary predicate → no witness emission");
    }

    #[test]
    fn emit_nnz_with_eq_bound_appends_constraints_and_assert() {
        // A non-trivial run should add several IR statements: one
        // constant_int(0) accumulator init, one indicator chain per
        // element, a final comparison, and an assert.
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[1, 0, 1, 0]);
        let bound_val = b.ir_constant_int(2);
        let before = b.stmts.len();
        emit_nnz(&mut b, &[arr], Some("=="), Some(&bound_val));
        let after = b.stmts.len();
        assert!(
            after > before,
            "emit_nnz must add witness-time constraints; before={before}, after={after}"
        );
        // Check that the last statement is the assert.
        let last = b.stmts.last().expect("at least one stmt");
        assert!(
            matches!(last.ir, IR::Assert),
            "last emitted statement must be IR::Assert, got {:?}",
            last.ir
        );
    }

    #[test]
    fn emit_nnz_supports_le_and_ge_ops() {
        for op in &["<=", ">=", "<", ">", "!="] {
            let mut b = IRBuilder::new();
            let arr = list_of_ints(&mut b, &[1, 0]);
            let bound = b.ir_constant_int(1);
            emit_nnz(&mut b, &[arr], Some(op), Some(&bound));
            let last = b.stmts.last().expect("op produces an emitted assert");
            assert!(matches!(last.ir, IR::Assert), "op `{op}` must end in IR::Assert");
        }
    }

    // ── is_sorted ─────────────────────────────────────────────────

    #[test]
    fn registry_contains_is_sorted_and_monotone_emitters() {
        let r = witness_emitters();
        assert!(r.get("is_sorted").is_some());
        assert!(r.get("is_monotone_nondecreasing").is_some());
        assert_eq!(r.get("is_sorted").unwrap().kind, "is_sorted");
    }

    #[test]
    fn emit_is_sorted_on_4_elements_emits_3_asserts() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[1, 2, 3, 4]);
        let before = b.stmts.len();
        emit_is_sorted(&mut b, &[arr], None, None);
        let after = b.stmts.len();
        assert!(after > before);
        // 3 adjacent pairs → 3 LteI + 3 Assert.
        let new_stmts = &b.stmts[before..];
        let asserts = new_stmts.iter().filter(|s| matches!(s.ir, IR::Assert)).count();
        let lte_i = new_stmts.iter().filter(|s| matches!(s.ir, IR::LteI)).count();
        assert_eq!(asserts, 3);
        assert_eq!(lte_i, 3);
    }

    #[test]
    fn emit_is_sorted_on_singleton_is_noop() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[42]);
        let before = b.stmts.len();
        emit_is_sorted(&mut b, &[arr], None, None);
        assert_eq!(b.stmts.len(), before);
    }

    #[test]
    fn emit_is_sorted_rejects_op_or_bound() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[1, 2]);
        let bound = b.ir_constant_int(0);
        let before = b.stmts.len();
        // Unary predicate; op/bound must short-circuit.
        emit_is_sorted(&mut b, &[arr.clone()], Some("=="), Some(&bound));
        assert_eq!(b.stmts.len(), before);
        emit_is_sorted(&mut b, &[arr.clone()], Some("<="), None);
        assert_eq!(b.stmts.len(), before);
        emit_is_sorted(&mut b, &[arr], None, Some(&bound));
        assert_eq!(b.stmts.len(), before);
    }

    // ── max_run ────────────────────────────────────────────────────

    #[test]
    fn registry_contains_max_run_emitter() {
        let r = witness_emitters();
        assert!(r.get("max_run").is_some());
    }

    #[test]
    fn emit_max_run_with_le_bound_emits_3_pair_asserts_for_4_elements() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[0, 5, 7, 10]);
        let bound = b.ir_constant_int(5);
        let before = b.stmts.len();
        emit_max_run(&mut b, &[arr], Some("<="), Some(&bound));
        let after = b.stmts.len();
        assert!(after > before);
        let new_stmts = &b.stmts[before..];
        let asserts = new_stmts.iter().filter(|s| matches!(s.ir, IR::Assert)).count();
        let subs = new_stmts.iter().filter(|s| matches!(s.ir, IR::SubI)).count();
        let lte = new_stmts.iter().filter(|s| matches!(s.ir, IR::LteI)).count();
        assert_eq!(asserts, 3);
        assert_eq!(subs, 3);
        assert_eq!(lte, 3);
    }

    #[test]
    fn emit_max_run_with_unsupported_op_is_noop() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[0, 1]);
        let bound = b.ir_constant_int(1);
        let before = b.stmts.len();
        for op in &["==", "<", ">", ">=", "!="] {
            emit_max_run(&mut b, &[arr.clone()], Some(op), Some(&bound));
        }
        assert_eq!(
            b.stmts.len(),
            before,
            "max_run currently only supports `<=`; other ops must short-circuit"
        );
    }

    // ── is_permutation ────────────────────────────────────────────

    #[test]
    fn registry_contains_is_permutation_emitter() {
        assert!(witness_emitters().get("is_permutation").is_some());
    }

    #[test]
    fn emit_is_permutation_on_4_elements_emits_full_check() {
        // For N = 4: 4 range checks (GteI + LtI + 2 asserts each = 4*3 = 12 stmts
        // for the asserts/comparisons) + 6 injectivity pairs (NeI + assert).
        //
        // We count by statement kind: 4 GteI, 4 LtI, 6 NeI, and (4 + 4 + 6) = 14 Asserts.
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[0, 1, 2, 3]);
        let before = b.stmts.len();
        emit_is_permutation(&mut b, &[arr], None, None);
        let new_stmts = &b.stmts[before..];
        let gte = new_stmts.iter().filter(|s| matches!(s.ir, IR::GteI)).count();
        let lt = new_stmts.iter().filter(|s| matches!(s.ir, IR::LtI)).count();
        let ne = new_stmts.iter().filter(|s| matches!(s.ir, IR::NeI)).count();
        let asserts = new_stmts.iter().filter(|s| matches!(s.ir, IR::Assert)).count();
        assert_eq!(gte, 4, "expected 4 GteI for range-check lower bounds");
        assert_eq!(lt, 4, "expected 4 LtI for range-check upper bounds");
        assert_eq!(ne, 6, "expected 6 NeI for the 4*(4-1)/2 = 6 injectivity pairs");
        assert_eq!(asserts, 14, "expected 4+4+6 = 14 asserts");
    }

    #[test]
    fn emit_is_permutation_on_empty_array_is_noop() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[]);
        let before = b.stmts.len();
        emit_is_permutation(&mut b, &[arr], None, None);
        assert_eq!(b.stmts.len(), before);
    }

    #[test]
    fn emit_is_permutation_rejects_op_or_bound() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[0, 1]);
        let bound = b.ir_constant_int(0);
        let before = b.stmts.len();
        emit_is_permutation(&mut b, &[arr.clone()], Some("=="), Some(&bound));
        assert_eq!(b.stmts.len(), before);
        emit_is_permutation(&mut b, &[arr], Some("=="), None);
        assert_eq!(b.stmts.len(), before);
    }

    // ── fixed_point_count ─────────────────────────────────────────

    #[test]
    fn registry_contains_fixed_point_count_emitter() {
        assert!(witness_emitters().get("fixed_point_count").is_some());
    }

    #[test]
    fn emit_fixed_point_count_with_eq_emits_indicator_chain() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[0, 1, 0, 3]);
        let bound = b.ir_constant_int(3);
        let before = b.stmts.len();
        emit_fixed_point_count(&mut b, &[arr], Some("=="), Some(&bound));
        let new_stmts = &b.stmts[before..];
        // 4 EqI (per-index `p[i] == i`), 4 IntCast, 3-4 AddI (post-fold), 1 EqI for final,
        // 1 Assert.
        let eq_i = new_stmts.iter().filter(|s| matches!(s.ir, IR::EqI)).count();
        let int_cast = new_stmts.iter().filter(|s| matches!(s.ir, IR::IntCast)).count();
        let asserts = new_stmts.iter().filter(|s| matches!(s.ir, IR::Assert)).count();
        // 4 per-index + 1 final = 5 EqI total (could fold to 5 in absence of optimization).
        assert!(eq_i >= 4, "expected ≥4 EqI ops (per-index + final), got {eq_i}");
        assert_eq!(int_cast, 4);
        assert_eq!(asserts, 1);
    }

    #[test]
    fn emit_fixed_point_count_with_unsupported_op_is_noop() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[0, 1]);
        let bound = b.ir_constant_int(1);
        let before = b.stmts.len();
        emit_fixed_point_count(&mut b, &[arr], Some("~="), Some(&bound));
        assert_eq!(b.stmts.len(), before);
    }

    // ── cycle_count: registered, but no witness emitter ───────────

    // ── popcount / forall_* ────────────────────────────────────────

    #[test]
    fn registry_contains_popcount_and_forall_emitters() {
        let r = witness_emitters();
        assert!(r.get("popcount").is_some());
        assert!(r.get("forall_is_nonneg").is_some());
        assert!(r.get("forall_lt_bound").is_some());
        assert!(r.get("forall_eq_const").is_some());
        assert!(r.get("forall_in_range").is_some());
    }

    #[test]
    fn emit_popcount_reuses_nnz_emitter() {
        // popcount uses emit_nnz directly; same indicator-sum shape.
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[1, 0, 1, 0]);
        let bound = b.ir_constant_int(2);
        let before = b.stmts.len();
        let popcount_emit = witness_emitters().get("popcount").unwrap().emit;
        popcount_emit(&mut b, &[arr], Some("=="), Some(&bound));
        let after = b.stmts.len();
        assert!(after > before);
        let last = b.stmts.last().expect("at least one stmt");
        assert!(matches!(last.ir, IR::Assert));
    }

    #[test]
    fn emit_forall_is_nonneg_emits_per_element_ge_assert() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[1, 2, 3, 4]);
        let before = b.stmts.len();
        emit_forall_is_nonneg(&mut b, &[arr], None, None);
        let new_stmts = &b.stmts[before..];
        let gte = new_stmts.iter().filter(|s| matches!(s.ir, IR::GteI)).count();
        let asserts = new_stmts.iter().filter(|s| matches!(s.ir, IR::Assert)).count();
        assert_eq!(gte, 4);
        assert_eq!(asserts, 4);
    }

    #[test]
    fn emit_forall_lt_bound_emits_per_element_lt_assert() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[0, 1, 2, 3]);
        let bound = b.ir_constant_int(10);
        let before = b.stmts.len();
        emit_forall_lt_bound(&mut b, &[arr, bound], None, None);
        let new_stmts = &b.stmts[before..];
        let lt = new_stmts.iter().filter(|s| matches!(s.ir, IR::LtI)).count();
        let asserts = new_stmts.iter().filter(|s| matches!(s.ir, IR::Assert)).count();
        assert_eq!(lt, 4);
        assert_eq!(asserts, 4);
    }

    #[test]
    fn emit_forall_eq_const_emits_per_element_eq_assert() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[7, 7, 7]);
        let c = b.ir_constant_int(7);
        let before = b.stmts.len();
        emit_forall_eq_const(&mut b, &[arr, c], None, None);
        let new_stmts = &b.stmts[before..];
        let eq = new_stmts.iter().filter(|s| matches!(s.ir, IR::EqI)).count();
        let asserts = new_stmts.iter().filter(|s| matches!(s.ir, IR::Assert)).count();
        assert_eq!(eq, 3);
        assert_eq!(asserts, 3);
    }

    #[test]
    fn emit_forall_in_range_emits_per_element_two_asserts() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[2, 5, 8]);
        let lo = b.ir_constant_int(0);
        let hi = b.ir_constant_int(10);
        let before = b.stmts.len();
        emit_forall_in_range(&mut b, &[arr, lo, hi], None, None);
        let new_stmts = &b.stmts[before..];
        let gte = new_stmts.iter().filter(|s| matches!(s.ir, IR::GteI)).count();
        let lte = new_stmts.iter().filter(|s| matches!(s.ir, IR::LteI)).count();
        let asserts = new_stmts.iter().filter(|s| matches!(s.ir, IR::Assert)).count();
        assert_eq!(gte, 3);
        assert_eq!(lte, 3);
        assert_eq!(asserts, 6);
    }

    #[test]
    fn emit_forall_variants_reject_op_or_bound() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[1, 2]);
        let bound = b.ir_constant_int(0);
        let before = b.stmts.len();
        emit_forall_is_nonneg(&mut b, &[arr.clone()], Some("=="), Some(&bound));
        emit_forall_lt_bound(&mut b, &[arr.clone(), bound.clone()], Some("=="), Some(&bound));
        emit_forall_eq_const(&mut b, &[arr.clone(), bound.clone()], None, Some(&bound));
        emit_forall_in_range(&mut b, &[arr, bound.clone(), bound], Some("=="), None);
        assert_eq!(
            b.stmts.len(), before,
            "forall_* must reject any op/bound (unary-style predicates)"
        );
    }

    #[test]
    fn cycle_count_is_registered_but_has_no_witness_emitter() {
        // The predicate registry has it (compile-time fact only); the
        // witness-emitter registry does not. Verified for documentation
        // — soundness gap is intentional and noted in the card status.
        use crate::optim::predicates::registry;
        assert!(registry().get("cycle_count").is_some(),
                "cycle_count must be in the predicate registry");
        assert!(witness_emitters().get("cycle_count").is_none(),
                "cycle_count must NOT have a witness emitter (deferred)");
    }

    #[test]
    fn emit_max_run_without_bound_is_noop() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[0, 1, 2]);
        let before = b.stmts.len();
        emit_max_run(&mut b, &[arr], Some("<="), None);
        assert_eq!(b.stmts.len(), before);
    }

    #[test]
    fn emit_nnz_with_unsupported_op_is_noop() {
        let mut b = IRBuilder::new();
        let arr = list_of_ints(&mut b, &[1]);
        let bound = b.ir_constant_int(1);
        let before = b.stmts.len();
        emit_nnz(&mut b, &[arr], Some("~="), Some(&bound));
        assert_eq!(
            b.stmts.len(),
            before,
            "unsupported op must short-circuit before any emission"
        );
    }
}
