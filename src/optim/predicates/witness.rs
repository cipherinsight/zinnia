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
//! or reject the program at compile time when it cannot (an unsupported
//! operator or operand, or a predicate with no in-circuit encoding).
//! Each emitter ships with a soundness paragraph in its body.
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
use crate::types::Value;

// ---------------------------------------------------------------------------
// WitnessEmitter
// ---------------------------------------------------------------------------

/// Per-predicate witness-time circuit emitter.
///
/// `emit` is called once per `ASTRequires` recognised at ir-gen time and
/// adds the constraints the prover must satisfy. The predicate is also a
/// compile-time fact the optimizer relies on, so a precondition the
/// emitter cannot enforce (an unsupported operator, operand or element
/// type) is a compile error, never a silent no-op: an unenforced fact
/// would let a malicious prover violate it and still produce a proof.
pub struct WitnessEmitter {
    pub kind: &'static str,
    pub emit: fn(&mut IRBuilder, &[Value], Option<&str>, Option<&Value>),
}

/// What an emitter returns: `Err(reason)` when it cannot enforce the
/// precondition as written.
type EmitResult = Result<(), String>;

/// Reject a precondition the circuit cannot enforce (a compile error).
fn reject_precondition(kind: &str, reason: &str) -> ! {
    panic!("@requires precondition `{kind}` cannot be enforced in the circuit: {reason}")
}

/// Wrap an `EmitResult`-returning emitter so an `Err` rejects the program.
macro_rules! enforcing_emitter {
    ($kind:literal, $emit:path) => {{
        fn emit_or_reject(
            b: &mut IRBuilder,
            args: &[Value],
            op: Option<&str>,
            bound: Option<&Value>,
        ) {
            if let Err(reason) = $emit(b, args, op, bound) {
                reject_precondition($kind, &reason);
            }
        }
        WitnessEmitter {
            kind: $kind,
            emit: emit_or_reject,
        }
    }};
}

/// Global registry — populated lazily. Every user-visible predicate kind
/// has an entry; `cycle_count`, which has no in-circuit encoding, is
/// registered with an emitter that rejects it.
pub fn witness_emitters() -> &'static HashMap<&'static str, WitnessEmitter> {
    static EMITTERS: OnceLock<HashMap<&'static str, WitnessEmitter>> = OnceLock::new();
    EMITTERS.get_or_init(build_witness_emitters)
}

fn build_witness_emitters() -> HashMap<&'static str, WitnessEmitter> {
    let emitters = [
        // Cardinality: indicator sum compared with the bound. `popcount`
        // is `nnz` on boolean arrays.
        enforcing_emitter!("nnz", emit_nnz),
        enforcing_emitter!("popcount", emit_nnz),
        // Ordering: per-adjacent-pair `arr[i] <= arr[i+1]`.
        enforcing_emitter!("is_sorted", emit_is_sorted),
        enforcing_emitter!("is_monotone_nondecreasing", emit_is_sorted),
        // Per-adjacent-pair gap `<=` bound.
        enforcing_emitter!("max_run", emit_max_run),
        // Permutations: every position hit exactly once.
        enforcing_emitter!("is_permutation", emit_is_permutation),
        enforcing_emitter!("fixed_point_count", emit_fixed_point_count),
        enforcing_emitter!("cycle_count", emit_cycle_count),
        // `forall(arr, P)` flattened to one atom per inner predicate.
        enforcing_emitter!("forall_is_nonneg", emit_forall_is_nonneg),
        enforcing_emitter!("forall_lt_bound", emit_forall_lt_bound),
        enforcing_emitter!("forall_eq_const", emit_forall_eq_const),
        enforcing_emitter!("forall_in_range", emit_forall_in_range),
    ];
    emitters.into_iter().map(|e| (e.kind, e)).collect()
}

// ---------------------------------------------------------------------------
// Cardinality: nnz / popcount
// ---------------------------------------------------------------------------

/// `nnz(x) op k`: count the nonzero elements in-circuit and assert
/// `count op k`.
///
/// Soundness: `count = sum_i [x[i] != 0]` is computed from `x` by
/// constraints (each indicator is a comparison result, 0 or 1), so a
/// witness with `nnz(x) op k` false fails the final assert. The count is
/// at most `len(x)`, far below the field size, so the sum cannot wrap.
fn emit_nnz(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    let comparison = comparison_ir(op)?;
    let bound = bound.ok_or("a comparison bound is required, e.g. `nnz(x) == k`")?;
    let [array] = args else {
        return Err(format!("expected one array argument, got {}", args.len()));
    };
    let elements = collect_elements(b, array)?;
    let mut count = b.ir_constant_int(0);
    for elem in &elements {
        let indicator = nonzero_indicator(b, elem)?;
        count = b.ir_add_i(&count, &indicator);
    }
    let holds = b.create_ir(&comparison, &[count, bound.clone()]);
    b.ir_assert(&holds);
    // At most `k` nonzero values (`<= k`, `== k`; `< k` is `<= k - 1`),
    // resting on the assert just made.
    let at_most = match (op, bound.int_val()) {
        (Some("<=") | Some("=="), Some(k)) => Some(k),
        (Some("<"), Some(k)) => k.checked_sub(1),
        _ => None,
    };
    if let Some(k) = at_most.and_then(|k| u32::try_from(k).ok()) {
        b.ir_array_fact(ArrayFactKind::CountBound { max_nonzero: k }, &elements, &[holds]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Ordering: is_sorted / is_monotone_nondecreasing / max_run
// ---------------------------------------------------------------------------

/// `is_sorted(arr)`: assert `arr[i] <= arr[i+1]` for every adjacent pair.
///
/// Soundness: an unsorted array has a violating adjacent pair, whose
/// assert fails. Arrays of length 0 or 1 are sorted; nothing to emit.
fn emit_is_sorted(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    let array = unary_array_argument(args, op, bound)?;
    let elements = collect_elements(b, array)?;
    let mut basis = Vec::new();
    for pair in elements.windows(2) {
        let ordered = less_or_equal(b, &pair[0], &pair[1])?;
        b.ir_assert(&ordered);
        basis.push(ordered);
    }
    // The array claim the asserts make (`lower::array_facts`), for elements
    // of one type (each pair then compares the elements themselves).
    if same_number_type(&elements) {
        b.ir_array_fact(ArrayFactKind::NonDecreasing, &elements, &basis);
    }
    Ok(())
}

/// `max_run(arr) <= K`: assert `arr[i+1] - arr[i] <= K` for every
/// adjacent pair (the largest gap is one of these differences).
///
/// Soundness: if the largest gap exceeds `K`, its pair's assert fails.
/// Float arrays compare the exact fixed-point difference with `K`
/// converted to Float (truncating the difference to an integer first
/// would accept gaps in `(K, K+1)`). Only the `<=` form is supported.
fn emit_max_run(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    if op != Some("<=") {
        return Err("only the form `max_run(arr) <= K` is supported".to_string());
    }
    let bound = bound.ok_or("a bound K is required")?;
    let [array] = args else {
        return Err(format!("expected one array argument, got {}", args.len()));
    };
    let elements = collect_elements(b, array)?;
    for pair in elements.windows(2) {
        let within = match (&pair[0], &pair[1]) {
            (Value::Integer(_), Value::Integer(_)) => {
                let gap = b.ir_sub_i(&pair[1], &pair[0]);
                b.ir_less_than_or_equal_i(&gap, bound)
            }
            (Value::Float(_), Value::Float(_)) => {
                let gap = b.ir_sub_f(&pair[1], &pair[0]);
                let bound_f = b.ir_float_cast(bound);
                b.ir_less_than_or_equal_f(&gap, &bound_f)
            }
            _ => return Err("elements must be Integer or Float".to_string()),
        };
        b.ir_assert(&within);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Permutations: is_permutation / fixed_point_count / cycle_count
// ---------------------------------------------------------------------------

/// `is_permutation(p)` for a length-N integer array: one `OpKind::
/// IsPermutation` placeholder, asserted. The array lowering layer proves it
/// (`lower::permutation`): by hit counts, every position `j` in `0..N` hit
/// exactly once, `sum_i [p[i] == j] == 1` (each `p[i]` hits at most one of
/// the distinct `j`, so N hits, one per `j`, mean `p` is a permutation of
/// `0..N`; the hits are the one-hot `[c[j] == i]` of the inverse
/// permutation `c` (`c[p] = arange(N)`), which a scatter by hits reuses),
/// or by a grand product at a verifier challenge on halo2-base. The range
/// `0 <= p[i] < N` and the permutation are recorded for the index checks
/// and scatters that follow (each relies on the fact the check proves).
fn emit_is_permutation(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    let array = unary_array_argument(args, op, bound)?;
    let elements = collect_elements(b, array)?;
    if !elements.iter().all(|e| matches!(e, Value::Integer(_))) {
        return Err("elements must be Integer".to_string());
    }
    // The array lowering layer checks it (`lower::permutation`: hit counts,
    // or a grand product at a verifier challenge on halo2-base). The empty
    // array is a permutation of nothing.
    if !elements.is_empty() {
        let holds = crate::lower::permutation::check_lane(b, &elements);
        b.ir_assert(&holds);
        b.ir_array_fact(ArrayFactKind::Permutation, &elements, &[holds]);
    }
    b.record_permutation(&elements);
    Ok(())
}

/// `fixed_point_count(p) op K`: count the indices with `p[i] == i` and
/// assert `count op K`.
///
/// Soundness: as for `nnz`, the count is computed from `p` by
/// constraints, so a witness violating `count op K` fails the assert.
fn emit_fixed_point_count(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    let comparison = comparison_ir(op)?;
    let bound = bound.ok_or("a comparison bound is required")?;
    let [array] = args else {
        return Err(format!("expected one array argument, got {}", args.len()));
    };
    let mut count = b.ir_constant_int(0);
    for (i, elem) in collect_elements(b, array)?.iter().enumerate() {
        if !matches!(elem, Value::Integer(_)) {
            return Err("elements must be Integer".to_string());
        }
        let index = b.ir_constant_int(i as i64);
        let is_fixed = b.ir_equal_i(elem, &index);
        let indicator = b.ir_int_cast(&is_fixed);
        count = b.ir_add_i(&count, &indicator);
    }
    let holds = b.create_ir(&comparison, &[count, bound.clone()]);
    b.ir_assert(&holds);
    Ok(())
}

/// `cycle_count(p) op K` has no in-circuit encoding (counting cycles
/// needs a graph traversal), so it cannot be enforced and is rejected.
fn emit_cycle_count(_: &mut IRBuilder, _: &[Value], _: Option<&str>, _: Option<&Value>) -> EmitResult {
    Err("counting permutation cycles has no in-circuit encoding".to_string())
}

// ---------------------------------------------------------------------------
// forall_* element predicates
// ---------------------------------------------------------------------------

/// `forall(arr, is_nonneg)`: assert `arr[i] >= 0` for every element.
fn emit_forall_is_nonneg(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    let array = unary_array_argument(args, op, bound)?;
    let elements = collect_elements(b, array)?;
    let mut basis = Vec::new();
    for elem in elements.iter().cloned() {
        let holds = match &elem {
            Value::Integer(_) | Value::Boolean(_) => {
                let zero = b.ir_constant_int(0);
                b.ir_greater_than_or_equal_i(&elem, &zero)
            }
            Value::Float(_) => {
                let zero = b.ir_constant_float(0.0);
                b.ir_greater_than_or_equal_f(&elem, &zero)
            }
            _ => return Err("elements must be numeric".to_string()),
        };
        b.ir_assert(&holds);
        basis.push(holds);
    }
    state_elements_within(b, &elements, Some(0), None, &basis);
    Ok(())
}

/// `forall(arr, lt_bound(B))`: assert `arr[i] < B` for every element.
fn emit_forall_lt_bound(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    let [array, limit] = element_predicate_arguments::<2>(args, op, bound)?;
    let elements = collect_elements(b, array)?;
    let mut basis = Vec::new();
    for elem in &elements {
        let holds = compare_element(b, CmpOp::Lt, elem, limit)?;
        b.ir_assert(&holds);
        basis.push(holds);
    }
    let below = constant_code(limit, &elements).and_then(|c| c.checked_sub(1));
    if below.is_some() {
        state_elements_within(b, &elements, None, below, &basis);
    }
    Ok(())
}

/// `forall(arr, eq_const(C))`: assert `arr[i] == C` for every element.
fn emit_forall_eq_const(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    let [array, constant] = element_predicate_arguments::<2>(args, op, bound)?;
    let elements = collect_elements(b, array)?;
    let mut basis = Vec::new();
    for elem in &elements {
        let holds = compare_element(b, CmpOp::Eq, elem, constant)?;
        b.ir_assert(&holds);
        basis.push(holds);
    }
    if let Some(c) = constant_code(constant, &elements) {
        state_elements_within(b, &elements, Some(c), Some(c), &basis);
    }
    Ok(())
}

/// `forall(arr, in_range(lo, hi))`: assert `lo <= arr[i] <= hi` for every
/// element.
fn emit_forall_in_range(b: &mut IRBuilder, args: &[Value], op: Option<&str>, bound: Option<&Value>) -> EmitResult {
    let [array, lo, hi] = element_predicate_arguments::<3>(args, op, bound)?;
    let elements = collect_elements(b, array)?;
    let mut basis = Vec::new();
    for elem in &elements {
        let above_lo = compare_element(b, CmpOp::Ge, elem, lo)?;
        b.ir_assert(&above_lo);
        let below_hi = compare_element(b, CmpOp::Le, elem, hi)?;
        b.ir_assert(&below_hi);
        basis.extend([above_lo, below_hi]);
    }
    if let (Some(lo), Some(hi)) = (constant_code(lo, &elements), constant_code(hi, &elements)) {
        state_elements_within(b, &elements, Some(lo), Some(hi), &basis);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The array facts the element predicates assert (`lower::array_facts`)
// ---------------------------------------------------------------------------

use crate::lower::array_facts::ArrayFactKind;

/// Whether every element has one number type (each comparison of the
/// element predicates is then of the element itself, not of a cast).
fn same_number_type(elements: &[Value]) -> bool {
    elements.iter().all(|e| matches!(e, Value::Integer(_))) || elements.iter().all(|e| matches!(e, Value::Float(_)))
}

/// The code a constant bound has against `elements`: the integer for
/// integer elements, the fixed-point code of its value for float ones
/// (the comparisons compare floats as their codes, and an integer bound is
/// converted exactly).
fn constant_code(bound: &Value, elements: &[Value]) -> Option<i64> {
    if !same_number_type(elements) {
        return None;
    }
    let float = elements.first().is_some_and(|e| matches!(e, Value::Float(_)));
    let value = bound.float_val().or_else(|| bound.int_val().map(|i| i as f64))?;
    if float {
        let precision = crate::prove::types::ProvingParams::default().precision_bits;
        i64::try_from(crate::prove::field::quantize(value, precision)).ok()
    } else {
        bound.int_val()
    }
}

/// State that every element lies in `[lo, hi]` (codes), resting on the
/// element predicate's asserts (`basis`).
fn state_elements_within(b: &mut IRBuilder, elements: &[Value], lo: Option<i64>, hi: Option<i64>, basis: &[Value]) {
    if same_number_type(elements) {
        b.ir_array_fact(ArrayFactKind::ElementsWithin { lo, hi }, elements, basis);
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

use crate::optim::predicates::formula::{ArithOp, BoolOp, CmpOp, ContractTerm, ContractVar};

/// Lower a user-written `@requires` precondition (a `ContractTerm` with
/// `Var(Input(name))` leaves, top-level Bool) into Zinnia IR. Returns
/// the `Value::Boolean` representing the predicate's runtime value;
/// the caller wraps in `ir_assert(...)` to enforce at proof time.
///
/// Errors carry a string diagnostic; the scalar-precondition flow rejects
/// the program on error, because the precondition is already a fact the
/// optimizer relies on and must not go unenforced.
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
            let ir = match (&l, &r) {
                (Value::Integer(_) | Value::Boolean(_), Value::Integer(_) | Value::Boolean(_)) => {
                    match op {
                        CmpOp::Eq => IR::EqI,
                        CmpOp::Ne => IR::NeI,
                        CmpOp::Lt => IR::LtI,
                        CmpOp::Le => IR::LteI,
                        CmpOp::Gt => IR::GtI,
                        CmpOp::Ge => IR::GteI,
                    }
                }
                (Value::Float(_), Value::Float(_)) => match op {
                    CmpOp::Eq => IR::EqF,
                    CmpOp::Ne => IR::NeF,
                    CmpOp::Lt => IR::LtF,
                    CmpOp::Le => IR::LteF,
                    CmpOp::Gt => IR::GtF,
                    CmpOp::Ge => IR::GteF,
                },
                _ => {
                    return Err(format!(
                        "scalar precondition comparison mixes integer and float values; \
                         int-typed parameters cannot use float literal bounds"
                    ));
                }
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
        | ContractTerm::Arith { .. } => Err(format!(
            "scalar precondition Bool-context expected a comparison or \
                 logical composition; got a numeric-typed term"
        )),
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

        ContractTerm::Var(ContractVar::Output) => {
            Err("scalar precondition cannot reference `Output` (a contract \
             template var; not valid in @requires)"
                .to_string())
        }

        ContractTerm::Var(ContractVar::Value(_)) => Err(
            "scalar precondition cannot reference `Value` (only contract \
             templates carry ValueId-anchored leaves; @requires terms \
             reach the witness emitter pre-substitution with Input(name) \
             leaves only)"
                .to_string(),
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
            // `/` is true division (a float), as in Python and in the
            // predicate's facts (`formula::lower`).
            if *op == ArithOp::Div {
                let (l, r) = (b.ir_float_cast(&l), b.ir_float_cast(&r));
                return Ok(b.ir_div_f(&l, &r));
            }
            let ir = match (&l, &r) {
                (Value::Integer(_) | Value::Boolean(_), Value::Integer(_) | Value::Boolean(_)) => {
                    match op {
                        ArithOp::Add => IR::AddI,
                        ArithOp::Sub => IR::SubI,
                        ArithOp::Mul => IR::MulI,
                        ArithOp::Div => unreachable!("true division is handled above"),
                        ArithOp::FloorDiv => IR::FloorDivI,
                        ArithOp::Mod => IR::ModI,
                        ArithOp::Pow => IR::PowI,
                    }
                }
                (Value::Float(_), Value::Float(_)) => match op {
                    ArithOp::Add => IR::AddF,
                    ArithOp::Sub => IR::SubF,
                    ArithOp::Mul => IR::MulF,
                    ArithOp::Div => IR::DivF,
                    ArithOp::Pow => IR::PowF,
                    ArithOp::FloorDiv => IR::FloorDivF,
                    ArithOp::Mod => IR::ModF,
                },
                _ => {
                    return Err(format!(
                        "scalar precondition arithmetic mixes integer and float values; \
                         int-typed parameters cannot use float literal bounds"
                    ));
                }
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

/// The comparison IR for a predicate operator (`nnz(x) op k`).
fn comparison_ir(op: Option<&str>) -> Result<IR, String> {
    Ok(match op {
        Some("==") => IR::EqI,
        Some("!=") => IR::NeI,
        Some("<") => IR::LtI,
        Some("<=") => IR::LteI,
        Some(">") => IR::GtI,
        Some(">=") => IR::GteI,
        Some(other) => return Err(format!("unsupported comparison `{other}`")),
        None => return Err("a comparison with a bound is required".to_string()),
    })
}

/// The single array argument of a unary predicate (no operator, no bound).
fn unary_array_argument<'a>(
    args: &'a [Value],
    op: Option<&str>,
    bound: Option<&Value>,
) -> Result<&'a Value, String> {
    let [array] = element_predicate_arguments::<1>(args, op, bound)?;
    Ok(array)
}

/// The `N` arguments of an element predicate, which takes no operator and
/// no bound.
fn element_predicate_arguments<'a, const N: usize>(
    args: &'a [Value],
    op: Option<&str>,
    bound: Option<&Value>,
) -> Result<[&'a Value; N], String> {
    if op.is_some() || bound.is_some() {
        return Err("this predicate takes no comparison".to_string());
    }
    let refs: Vec<&Value> = args.iter().collect();
    refs.try_into()
        .map_err(|_| format!("expected {N} arguments, got {}", args.len()))
}

/// The elements of an array-like Value, flattened in row-major order (a
/// multi-dimensional array contributes every cell; the old version kept
/// only the rows and skipped them, leaving the predicate unchecked).
fn collect_elements(b: &mut IRBuilder, val: &Value) -> Result<Vec<Value>, String> {
    if !matches!(val, Value::StaticArray { .. } | Value::List(_) | Value::Tuple(_)) {
        return Err("the argument must be a static array or list".to_string());
    }
    let cells = crate::helpers::composite::flatten_composite_with_builder(b, val);
    if !cells.iter().all(Value::is_number) {
        return Err("only arrays of Integer, Float or Boolean are supported".to_string());
    }
    Ok(cells)
}

/// `lhs <= rhs` for two elements of the same numeric type.
fn less_or_equal(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Result<Value, String> {
    compare_element(b, CmpOp::Le, lhs, rhs)
}

/// `elem op other`, comparing in Float when either side is a Float.
fn compare_element(b: &mut IRBuilder, op: CmpOp, elem: &Value, other: &Value) -> Result<Value, String> {
    if !elem.is_number() || !other.is_number() {
        return Err("elements and bounds must be numeric".to_string());
    }
    let use_float = matches!(elem, Value::Float(_)) || matches!(other, Value::Float(_));
    Ok(if use_float {
        let l = b.ir_float_cast(elem);
        let r = b.ir_float_cast(other);
        match op {
            CmpOp::Eq => b.ir_equal_f(&l, &r),
            CmpOp::Ne => b.ir_not_equal_f(&l, &r),
            CmpOp::Lt => b.ir_less_than_f(&l, &r),
            CmpOp::Le => b.ir_less_than_or_equal_f(&l, &r),
            CmpOp::Gt => b.ir_greater_than_f(&l, &r),
            CmpOp::Ge => b.ir_greater_than_or_equal_f(&l, &r),
        }
    } else {
        let l = b.ir_int_cast(elem);
        let r = b.ir_int_cast(other);
        match op {
            CmpOp::Eq => b.ir_equal_i(&l, &r),
            CmpOp::Ne => b.ir_not_equal_i(&l, &r),
            CmpOp::Lt => b.ir_less_than_i(&l, &r),
            CmpOp::Le => b.ir_less_than_or_equal_i(&l, &r),
            CmpOp::Gt => b.ir_greater_than_i(&l, &r),
            CmpOp::Ge => b.ir_greater_than_or_equal_i(&l, &r),
        }
    })
}

/// A 0/1 Integer indicator for `elem != 0`.
///
/// The comparison result is a Boolean cell holding 0 or 1;
/// `IRBuilder::ir_int_cast` reinterprets it as an Integer without IR. (A
/// raw `IR::IntCast` would be wrong here: it is the Float→Integer
/// conversion and maps the Boolean cell 1 to 0.)
fn nonzero_indicator(b: &mut IRBuilder, elem: &Value) -> Result<Value, String> {
    let nonzero = match elem {
        Value::Boolean(_) => elem.clone(),
        Value::Integer(_) | Value::Float(_) => {
            let zero = b.ir_constant_int(0);
            compare_element(b, CmpOp::Ne, elem, &zero)?
        }
        _ => return Err("elements must be Integer, Float or Boolean".to_string()),
    };
    Ok(b.ir_int_cast(&nonzero))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::{CircuitInputs, ResolvedWitness};
    use crate::prove::mock::MockProverBackend;
    use crate::prove::traits::ProverBackend;
    use crate::prove::types::ProvingParams;
    use crate::types::{CompositeData, ValueId};

    fn list_of(values: Vec<Value>) -> Value {
        Value::List(CompositeData {
            elements_type: values.iter().map(|v| v.zinnia_type()).collect(),
            values,
            value_id: ValueId::next(),
        })
    }

    fn ints(b: &mut IRBuilder, values: &[i64]) -> Value {
        list_of(values.iter().map(|&n| b.ir_constant_int(n)).collect())
    }

    fn floats(b: &mut IRBuilder, values: &[f64]) -> Value {
        list_of(values.iter().map(|&x| b.ir_constant_float(x)).collect())
    }

    /// Emit `kind` on fresh constants and mock-prove the result: `true`
    /// iff the emitted constraints hold.
    fn holds(
        kind: &str,
        build: impl FnOnce(&mut IRBuilder) -> (Vec<Value>, Option<&'static str>, Option<Value>),
    ) -> bool {
        let mut b = IRBuilder::new();
        let (args, op, bound) = build(&mut b);
        let emitter = witness_emitters().get(kind).expect("registered");
        (emitter.emit)(&mut b, &args, op, bound.as_ref());
        // The array lowering layer lowers the placeholders some emitters
        // leave (`is_permutation`), as the compile pipeline does.
        let lowering = crate::lower::ArrayLowering { target: crate::lower::Target::Halo2Base };
        let graph = crate::optim::IRPass::exec(&lowering, b.export_ir_graph());
        let params = ProvingParams::default();
        let witness = ResolvedWitness::new(CircuitInputs::new(), params.precision_bits);
        let artifact = MockProverBackend.prove(&graph, &witness, &params).unwrap();
        artifact.proof_bytes == "mock_satisfied"
    }

    fn rejected(
        kind: &str,
        build: impl FnOnce(&mut IRBuilder) -> (Vec<Value>, Option<&'static str>, Option<Value>),
    ) -> bool {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| holds(kind, build))).is_err()
    }

    #[test]
    fn every_user_predicate_kind_has_an_emitter() {
        for kind in [
            "nnz", "popcount", "is_sorted", "is_monotone_nondecreasing", "max_run",
            "is_permutation", "cycle_count", "fixed_point_count", "forall_is_nonneg",
            "forall_lt_bound", "forall_eq_const", "forall_in_range",
        ] {
            assert_eq!(witness_emitters().get(kind).map(|e| e.kind), Some(kind));
        }
    }

    #[test]
    fn nnz_counts_nonzero_elements() {
        // The count used to be built with a raw Float->Integer `IntCast`
        // of each Boolean indicator, which maps 1 to 0: the honest
        // `nnz == 3` was rejected and the false `nnz == 0` accepted.
        let nnz_eq = |k: i64| move |b: &mut IRBuilder| {
            let x = ints(b, &[1, 0, 2, 3]);
            (vec![x], Some("=="), Some(b.ir_constant_int(k)))
        };
        assert!(holds("nnz", nnz_eq(3)));
        assert!(!holds("nnz", nnz_eq(0)));
        assert!(!holds("nnz", nnz_eq(4)));
        assert!(holds("nnz", |b| {
            let x = floats(b, &[0.5, 0.0, -2.0]);
            (vec![x], Some("<="), Some(b.ir_constant_int(2)))
        }));
        assert!(holds("popcount", |b| {
            let x = list_of(vec![b.ir_constant_bool(true), b.ir_constant_bool(false)]);
            (vec![x], Some("=="), Some(b.ir_constant_int(1)))
        }));
    }

    #[test]
    fn ordering_predicates() {
        assert!(holds("is_sorted", |b| (vec![ints(b, &[1, 2, 2, 7])], None, None)));
        assert!(!holds("is_sorted", |b| (vec![ints(b, &[1, 3, 2])], None, None)));
        assert!(holds("is_sorted", |b| (vec![ints(b, &[5])], None, None)));
        let max_run = |values: &'static [f64], k: i64| move |b: &mut IRBuilder| {
            (vec![floats(b, values)], Some("<="), Some(b.ir_constant_int(k)))
        };
        assert!(holds("max_run", max_run(&[0.0, 2.0, 4.0], 2)));
        // A gap of 2.5 exceeds K = 2; truncating it to 2 accepted it.
        assert!(!holds("max_run", max_run(&[0.0, 2.5], 2)));
    }

    #[test]
    fn permutation_predicates() {
        assert!(holds("is_permutation", |b| (vec![ints(b, &[2, 0, 1])], None, None)));
        assert!(!holds("is_permutation", |b| (vec![ints(b, &[2, 2, 1])], None, None)));
        assert!(!holds("is_permutation", |b| (vec![ints(b, &[0, 1, 3])], None, None)));
        let fixed_points = |k: i64| move |b: &mut IRBuilder| {
            (vec![ints(b, &[0, 2, 1, 3])], Some("=="), Some(b.ir_constant_int(k)))
        };
        assert!(holds("fixed_point_count", fixed_points(2)));
        assert!(!holds("fixed_point_count", fixed_points(0)));
    }

    #[test]
    fn forall_predicates() {
        assert!(holds("forall_is_nonneg", |b| (vec![ints(b, &[0, 3])], None, None)));
        assert!(!holds("forall_is_nonneg", |b| (vec![ints(b, &[0, -3])], None, None)));
        assert!(holds("forall_lt_bound", |b| {
            let x = ints(b, &[0, 9]);
            (vec![x, b.ir_constant_int(10)], None, None)
        }));
        assert!(!holds("forall_lt_bound", |b| {
            let x = ints(b, &[0, 10]);
            (vec![x, b.ir_constant_int(10)], None, None)
        }));
        assert!(holds("forall_eq_const", |b| {
            let x = floats(b, &[1.5, 1.5]);
            (vec![x, b.ir_constant_float(1.5)], None, None)
        }));
        assert!(!holds("forall_in_range", |b| {
            let x = ints(b, &[2, 6]);
            (vec![x, b.ir_constant_int(0), b.ir_constant_int(5)], None, None)
        }));
    }

    #[test]
    fn unenforceable_forms_are_rejected() {
        // Each of these used to return without emitting anything while the
        // optimizer still assumed the predicate.
        assert!(rejected("nnz", |b| (vec![ints(b, &[1])], Some("~="), Some(b.ir_constant_int(1)))));
        assert!(rejected("nnz", |b| (vec![ints(b, &[1])], None, None)));
        assert!(rejected("nnz", |b| (vec![b.ir_constant_int(1)], Some("=="), Some(b.ir_constant_int(1)))));
        assert!(rejected("max_run", |b| (vec![ints(b, &[1, 2])], Some("=="), Some(b.ir_constant_int(1)))));
        assert!(rejected("is_sorted", |b| (vec![ints(b, &[1, 2])], Some("=="), Some(b.ir_constant_int(0)))));
        assert!(rejected("is_permutation", |b| (vec![floats(b, &[0.0, 1.0])], None, None)));
        assert!(rejected("forall_in_range", |b| (vec![ints(b, &[1])], None, None)));
        assert!(rejected("cycle_count", |b| (vec![ints(b, &[1, 0])], Some("=="), Some(b.ir_constant_int(1)))));
    }
}
