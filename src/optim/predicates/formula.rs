//! `ContractTerm` — the typed AST for SMT contract templates.
//!
//! A contract template is a Boolean formula expressed against *formal*
//! variables (parameter names of the op being contracted). At chokepoint
//! discharge time, the formal vars are substituted with the actual SSA
//! arguments at the call site, and the resulting term is lowered to a
//! `z3::ast::Bool` and asserted on the solver.
//!
//! ## Why a typed AST instead of opaque strings?
//!
//! - **Soundness review**: a structured AST is easier to audit per-arm
//!   than a parser. Each `ContractTerm` variant has one well-defined
//!   meaning; reviewers do not need to verify a string-parser is
//!   round-trip-correct.
//! - **Substitution is straightforward**: walk the AST, swap leaves.
//! - **Per-predicate extension**: per-predicate cards add new
//!   `PredicateApp` cases (already keyed by string), and may extend
//!   `ContractTerm` itself with new variants behind a feature flag if a
//!   predicate needs a richer formula language.
//! - **Independent of the underlying solver**: lowering to `z3::Bool`
//!   lives in one place, so swapping solvers (e.g., to bitwuzla) is a
//!   single-file change.
//!
//! ## Granularity
//!
//! `ContractTerm` covers what the contracts framework needs to express
//! today: predicate applications (`nnz(x)`), comparisons (`<`, `<=`,
//! `==`, …), simple arithmetic (`+`, `-`, `*`), references to op inputs
//! / outputs, integer literals, and Boolean composition (`and`, `or`,
//! `not`). It deliberately does not cover quantifiers, arrays, or
//! function definitions — meta-facts handle quantified content via the
//! registry's `meta_facts_fn`, and the contract-template language stays
//! quantifier-free for tractability.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use z3::ast::{Bool, Int, Real};

use crate::optim::smt_encoding::Z3Term;
use crate::types::ValueId;

/// Bit-pattern wrapper around `f64` so float literals can live in
/// `ContractTerm` and still implement `Hash` / `Eq` (the derive on
/// `ContractTerm` needs both). NaN bit patterns compare equal to
/// themselves under this wrapper — contracts don't reason about NaN
/// arithmetic semantics anyway.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ContractFloat(pub f64);

impl PartialEq for ContractFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for ContractFloat {}

impl std::hash::Hash for ContractFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for ContractFloat {
    fn from(f: f64) -> Self {
        ContractFloat(f)
    }
}

/// A reference to a formal variable inside a contract template.
///
/// `Input(name)` and `Output` are the building blocks; later passes may
/// introduce `LengthOf` / `IndexOf` style sub-references but those are
/// out of scope for the foundation contracts card and would land
/// alongside per-predicate cards that need them.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContractVar {
    /// A named chip input — the actual function-parameter name as the user
    /// wrote it. Used by user `@requires` terms; resolved at lowering time
    /// by the resolver via [`Substitution::with_input`]. Survives contract
    /// instantiation unchanged.
    Input(String),
    /// A named **template formal** of an op contract (e.g., `arg0`, `len`).
    /// Replaced with a concrete `Value(ValueId)` by [`instantiate_contract`]
    /// at the call site. Reaching the lowering layer with a `Formal` leaf
    /// is a template-shape bug.
    Formal(String),
    /// The op's output value (the result). A template-side placeholder
    /// like `Formal`, but with a single canonical name. Substituted to
    /// `Value(result_value_id)` at instantiation time.
    Output,
    /// Reference to a concrete compilation-layer Value by its `ValueId`.
    /// (compiler.value-id-and-fact-leaves) Replaces the previous
    /// `SsaPtr(StmtId)` leaf — facts now speak ValueId, not stmt_id. The
    /// witness emitter is the single bridge that maps `ValueId → StmtId`
    /// when it has to re-enter the IR layer.
    Value(ValueId),
}

/// Comparison operators expressible in contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Arithmetic operators expressible in contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    /// Integer / Float division. For Int, uses Zinnia's `DivI` (field
    /// inverse semantics within the prime modulus); for Float, `DivF`.
    Div,
    /// Floor division. `FloorDivI` for Int.
    FloorDiv,
    /// Modulo. `ModI` for Int.
    Mod,
    /// Power. SMT-unfriendly for non-constant exponents; the resolver
    /// timeout protects against pathological cases.
    Pow,
}

/// Boolean composition operators expressible in contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoolOp {
    And,
    Or,
}

/// A contract-template term. The AST is intentionally small.
///
/// Numeric subterms (`LitInt` / `LitFloat` / `Arith` / `Var`) lower to
/// `Z3Term::Int` or `Z3Term::Real`; sort is inferred from the operands.
/// Mixed-sort `Arith` and `Cmp` coerce ints to reals (Z3 `Int::to_real`)
/// so a contract can freely mix `LitInt(5)` and `LitFloat(0.5)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContractTerm {
    // ── Leaves ────────────────────────────────────────────────────
    /// Reference to a formal variable (input/output of the op).
    Var(ContractVar),
    /// Integer literal.
    LitInt(i64),
    /// Float literal. (compiler.float-contracts)
    LitFloat(ContractFloat),

    // ── Arithmetic (numeric result; sort follows operands) ───────
    Arith {
        op: ArithOp,
        lhs: Box<ContractTerm>,
        rhs: Box<ContractTerm>,
    },

    // ── Predicate application (Bool-typed result) ─────────────────
    /// A registered structural predicate applied to `args`. The kind
    /// must be present in [`crate::optim::predicates::registry()`].
    PredicateApp {
        kind: String,
        args: Vec<ContractTerm>,
    },

    // ── Comparison / Boolean composition (Bool-typed result) ──────
    Cmp {
        op: CmpOp,
        lhs: Box<ContractTerm>,
        rhs: Box<ContractTerm>,
    },
    BoolComb {
        op: BoolOp,
        operands: Vec<ContractTerm>,
    },
    Not(Box<ContractTerm>),
    LitBool(bool),
}

/// Builder helpers — preferred over hand-construction at contract sites.
impl ContractTerm {
    pub fn var_in(name: impl Into<String>) -> Self {
        ContractTerm::Var(ContractVar::Input(name.into()))
    }

    pub fn var_out() -> Self {
        ContractTerm::Var(ContractVar::Output)
    }

    pub fn lit(n: i64) -> Self {
        ContractTerm::LitInt(n)
    }

    pub fn lit_float(f: f64) -> Self {
        ContractTerm::LitFloat(ContractFloat(f))
    }

    pub fn cmp(op: CmpOp, lhs: ContractTerm, rhs: ContractTerm) -> Self {
        ContractTerm::Cmp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn eq(lhs: ContractTerm, rhs: ContractTerm) -> Self {
        Self::cmp(CmpOp::Eq, lhs, rhs)
    }

    pub fn le(lhs: ContractTerm, rhs: ContractTerm) -> Self {
        Self::cmp(CmpOp::Le, lhs, rhs)
    }

    pub fn ge(lhs: ContractTerm, rhs: ContractTerm) -> Self {
        Self::cmp(CmpOp::Ge, lhs, rhs)
    }

    pub fn pred(kind: impl Into<String>, args: Vec<ContractTerm>) -> Self {
        ContractTerm::PredicateApp {
            kind: kind.into(),
            args,
        }
    }

    pub fn and(operands: Vec<ContractTerm>) -> Self {
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands,
        }
    }

    pub fn add(lhs: ContractTerm, rhs: ContractTerm) -> Self {
        ContractTerm::Arith {
            op: ArithOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

// ---------------------------------------------------------------------------
// Substitution: formal-var name → concrete Z3 Int term
// ---------------------------------------------------------------------------

/// A mapping from formal `ContractVar` names to concrete `Z3Term`s.
/// Each entry can be `Int`, `Bool`, or `Real`; the lowering layer
/// dispatches `Cmp` / `Arith` on operand sorts. Old `with_input(name,
/// Int)` etc. still work — the API auto-wraps as `Z3Term::Int`.
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    /// Map: input name → its `Z3Term` at the call site.
    inputs: HashMap<String, Z3Term>,
    /// The op's output term, if known.
    output: Option<Z3Term>,
    /// Map: `ValueId` → its `Z3Term`. Populated by the fact-propagation
    /// framework when lowering value-anchored facts (`Var(Value(_))`).
    value_ids: HashMap<ValueId, Z3Term>,
}

impl Substitution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_input(mut self, name: impl Into<String>, term: Int) -> Self {
        self.inputs.insert(name.into(), Z3Term::Int(term));
        self
    }

    pub fn with_input_term(mut self, name: impl Into<String>, term: Z3Term) -> Self {
        self.inputs.insert(name.into(), term);
        self
    }

    pub fn with_output(mut self, term: Int) -> Self {
        self.output = Some(Z3Term::Int(term));
        self
    }

    pub fn with_output_term(mut self, term: Z3Term) -> Self {
        self.output = Some(term);
        self
    }

    pub fn with_value_id(mut self, vid: ValueId, term: Int) -> Self {
        self.value_ids.insert(vid, Z3Term::Int(term));
        self
    }

    pub fn with_value_id_term(mut self, vid: ValueId, term: Z3Term) -> Self {
        self.value_ids.insert(vid, term);
        self
    }

    pub fn input(&self, name: &str) -> Option<&Z3Term> {
        self.inputs.get(name)
    }

    pub fn output(&self) -> Option<&Z3Term> {
        self.output.as_ref()
    }

    pub fn value_id(&self, vid: ValueId) -> Option<&Z3Term> {
        self.value_ids.get(&vid)
    }
}

// ---------------------------------------------------------------------------
// Z3 lowering
// ---------------------------------------------------------------------------

/// Errors that may arise when lowering a `ContractTerm` to Z3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// A `Var(Input(name))` referenced a name not in the substitution.
    UnboundInput(String),
    /// A `Var(Output)` was referenced but the substitution has no output.
    UnboundOutput,
    /// A `Var(Value(vid))` referenced a ValueId not in the substitution.
    UnboundValueId(ValueId),
    /// A `Var(Formal(name))` survived to the lowering layer. Contract
    /// templates must be fully substituted by `instantiate_contract`
    /// before lowering; reaching Z3 with a formal is a template-shape bug.
    UnboundFormal(String),
    /// A `PredicateApp { kind, .. }` references an unregistered predicate.
    UnknownPredicate(String),
    /// Arity mismatch on a `PredicateApp`.
    ArityMismatch { kind: String, expected: usize, got: usize },
    /// A subterm that should have lowered to `Int` produced `Bool` (or vice
    /// versa). Indicates a malformed contract template; reviewers should
    /// catch this at audit time.
    SortMismatch(&'static str),
}

/// Output of lowering: a typed Z3 term plus any meta-facts whose injection
/// the lowering accumulated as a side effect (e.g., predicate references
/// that triggered the registry's `meta_facts_fn`).
///
/// Callers typically discard `meta_facts` (the SmtEncodingCtx is the proper
/// home for cross-formula dedup) but the test harness needs them visible.
#[derive(Debug, Clone)]
pub struct LowerOutput {
    /// The lowered Z3 Bool (top-level of a contract term must be Bool).
    pub term: Bool,
    /// Meta-facts accumulated during lowering. Each entry is a
    /// `Vec<Bool>` from one predicate's `meta_facts_fn` invocation.
    pub meta_fact_sets: Vec<(String, Vec<Bool>)>,
}

/// Lower a `ContractTerm` whose top-level must produce a `Bool`. Internal
/// arithmetic terms produce `Int`. The split between
/// [`lower_bool`] and [`lower_int`] enforces sort discipline statically.
pub fn lower_bool(
    term: &ContractTerm,
    subst: &Substitution,
) -> Result<LowerOutput, LowerError> {
    let mut meta_fact_sets = Vec::new();
    let bool_term = lower_bool_inner(term, subst, &mut meta_fact_sets)?;
    Ok(LowerOutput {
        term: bool_term,
        meta_fact_sets,
    })
}

fn lower_bool_inner(
    term: &ContractTerm,
    subst: &Substitution,
    meta_fact_sets: &mut Vec<(String, Vec<Bool>)>,
) -> Result<Bool, LowerError> {
    match term {
        ContractTerm::LitBool(b) => Ok(Bool::from_bool(*b)),

        ContractTerm::Cmp { op, lhs, rhs } => {
            let l = lower_num_inner(lhs, subst, meta_fact_sets)?;
            let r = lower_num_inner(rhs, subst, meta_fact_sets)?;
            // Float-aware dispatch (compiler.float-contracts): if either
            // side is Real, coerce both to Real and use Real comparisons;
            // otherwise both stay Int. Bool operands coerce to whichever
            // sort the other operand is.
            let any_real = matches!(l, Z3Term::Real(_)) || matches!(r, Z3Term::Real(_));
            Ok(if any_real {
                let lr = l.as_real();
                let rr = r.as_real();
                match op {
                    CmpOp::Eq => lr.eq(&rr),
                    CmpOp::Ne => lr.eq(&rr).not(),
                    CmpOp::Lt => lr.lt(&rr),
                    CmpOp::Le => lr.le(&rr),
                    CmpOp::Gt => lr.gt(&rr),
                    CmpOp::Ge => lr.ge(&rr),
                }
            } else {
                let li = l.as_int();
                let ri = r.as_int();
                match op {
                    CmpOp::Eq => li.eq(&ri),
                    CmpOp::Ne => li.eq(&ri).not(),
                    CmpOp::Lt => li.lt(&ri),
                    CmpOp::Le => li.le(&ri),
                    CmpOp::Gt => li.gt(&ri),
                    CmpOp::Ge => li.ge(&ri),
                }
            })
        }

        ContractTerm::BoolComb { op, operands } => {
            let lowered: Result<Vec<Bool>, _> = operands
                .iter()
                .map(|o| lower_bool_inner(o, subst, meta_fact_sets))
                .collect();
            let lowered = lowered?;
            let refs: Vec<&Bool> = lowered.iter().collect();
            Ok(match op {
                BoolOp::And => Bool::and(&refs),
                BoolOp::Or => Bool::or(&refs),
            })
        }

        ContractTerm::Not(inner) => {
            Ok(lower_bool_inner(inner, subst, meta_fact_sets)?.not())
        }

        ContractTerm::PredicateApp { kind, args } => {
            let registry = crate::optim::predicates::registry::registry();
            let reg = registry
                .get(kind.as_str())
                .ok_or_else(|| LowerError::UnknownPredicate(kind.clone()))?;
            if args.len() != reg.arity {
                return Err(LowerError::ArityMismatch {
                    kind: kind.clone(),
                    expected: reg.arity,
                    got: args.len(),
                });
            }
            let int_args: Result<Vec<Int>, _> = args
                .iter()
                .map(|a| lower_int_inner(a, subst, meta_fact_sets))
                .collect();
            let int_args = int_args?;
            // Collect meta-facts for the caller to inject. (Encoding-context
            // dedup happens at the SmtEncodingCtx layer, not here — that
            // way one ContractTerm can reference the same predicate twice
            // and we record both pre-dedup, letting the ctx decide.)
            let facts = reg.meta_facts(&int_args);
            if !facts.is_empty() {
                meta_fact_sets.push((kind.clone(), facts));
            }
            Ok(reg.encode_app(&int_args))
        }

        // Pure numeric leaves at the top of a Bool position is a
        // template-shape bug. Surface immediately rather than coercing.
        ContractTerm::Var(_)
        | ContractTerm::LitInt(_)
        | ContractTerm::LitFloat(_)
        | ContractTerm::Arith { .. } => Err(LowerError::SortMismatch(
            "Bool-context expected; got numeric-typed term",
        )),
    }
}

/// Lower a numeric subterm (Int or Real). Returns a `Z3Term` so the
/// caller can dispatch on sort. Bool-typed terms are rejected.
fn lower_num_inner(
    term: &ContractTerm,
    subst: &Substitution,
    meta_fact_sets: &mut Vec<(String, Vec<Bool>)>,
) -> Result<Z3Term, LowerError> {
    match term {
        ContractTerm::Var(ContractVar::Input(name)) => subst
            .input(name)
            .cloned()
            .ok_or_else(|| LowerError::UnboundInput(name.clone())),

        ContractTerm::Var(ContractVar::Output) => subst
            .output()
            .cloned()
            .ok_or(LowerError::UnboundOutput),

        ContractTerm::Var(ContractVar::Value(vid)) => subst
            .value_id(*vid)
            .cloned()
            .ok_or(LowerError::UnboundValueId(*vid)),

        ContractTerm::Var(ContractVar::Formal(name)) => {
            Err(LowerError::UnboundFormal(name.clone()))
        }

        ContractTerm::LitInt(n) => Ok(Z3Term::Int(Int::from_i64(*n))),
        ContractTerm::LitFloat(f) => Ok(Z3Term::Real(f64_to_z3_real(f.0))),

        ContractTerm::Arith { op, lhs, rhs } => {
            let l = lower_num_inner(lhs, subst, meta_fact_sets)?;
            let r = lower_num_inner(rhs, subst, meta_fact_sets)?;
            let any_real = matches!(l, Z3Term::Real(_)) || matches!(r, Z3Term::Real(_));
            if any_real {
                let lr = l.as_real();
                let rr = r.as_real();
                Ok(Z3Term::Real(match op {
                    ArithOp::Add => Real::add(&[&lr, &rr]),
                    ArithOp::Sub => Real::sub(&[&lr, &rr]),
                    ArithOp::Mul => Real::mul(&[&lr, &rr]),
                    ArithOp::Div => lr.div(&rr),
                    // Real floor-div lowers via to_int / from_int round-trip.
                    ArithOp::FloorDiv => Real::from_int(&lr.div(&rr).to_int()),
                    // Modulo and Pow on Real are not directly representable;
                    // emit a fresh Real (sound, imprecise).
                    ArithOp::Mod | ArithOp::Pow => Real::fresh_const("real_arith_unsup_"),
                }))
            } else {
                let li = l.as_int();
                let ri = r.as_int();
                Ok(Z3Term::Int(match op {
                    ArithOp::Add => Int::add(&[&li, &ri]),
                    ArithOp::Sub => Int::sub(&[&li, &ri]),
                    ArithOp::Mul => Int::mul(&[&li, &ri]),
                    ArithOp::Div | ArithOp::FloorDiv => li.div(&ri),
                    ArithOp::Mod => li.modulo(&ri),
                    // Pow: fresh Int (sound, imprecise) — same as pre-Phase
                    // float-contracts behaviour.
                    ArithOp::Pow => Int::fresh_const("pow_result_"),
                }))
            }
        }

        // Bool-typed nodes are template-shape bugs in a numeric position.
        _ => Err(LowerError::SortMismatch(
            "numeric-context expected; got Bool-typed term",
        )),
    }
}

/// Lower an `f64` to a Z3 `Real` exactly when the decimal expansion is
/// finite. Uses Rust's `f64::to_string` (which prints the shortest
/// round-trippable decimal) and splits into `num/den` where `den` is a
/// power of 10. Non-finite inputs (NaN / inf) collapse to a fresh
/// unconstrained Real (sound: they can't be reasoned about anyway).
fn f64_to_z3_real(f: f64) -> Real {
    if !f.is_finite() {
        return Real::fresh_const("nonfinite_lit_");
    }
    let s = format!("{}", f);
    // Strip an optional scientific-notation suffix like "1e-5". Rust's
    // default `f64::Display` shouldn't emit scientific for typical
    // values but we guard just in case.
    if s.contains('e') || s.contains('E') {
        // Fallback: lossless via bit-pattern is involved; for now,
        // accept slight imprecision via the decimal cast.
        return Real::from_rational_str(&format!("{}", f as i64), "1")
            .unwrap_or_else(|| Real::fresh_const("sci_notation_lit_"));
    }
    let (num_str, den_str) = if let Some(dot_idx) = s.find('.') {
        let int_part = &s[..dot_idx];
        let frac_part = &s[dot_idx + 1..];
        let den: String = std::iter::once('1')
            .chain(std::iter::repeat('0').take(frac_part.len()))
            .collect();
        let num = format!("{}{}", int_part, frac_part);
        (num, den)
    } else {
        (s, "1".to_string())
    };
    Real::from_rational_str(&num_str, &den_str)
        .unwrap_or_else(|| Real::fresh_const("malformed_lit_"))
}

/// Backwards-compatible helper: lower a numeric subterm and coerce to
/// `Int`. Used by callers that fundamentally expect Int (`PredicateApp`
/// argument lowering, which feeds predicate registries that take Int
/// args). Float values get coerced via `Real::to_int` (floor toward zero).
fn lower_int_inner(
    term: &ContractTerm,
    subst: &Substitution,
    meta_fact_sets: &mut Vec<(String, Vec<Bool>)>,
) -> Result<Int, LowerError> {
    Ok(lower_num_inner(term, subst, meta_fact_sets)?.as_int())
}

// ---------------------------------------------------------------------------
// Tests — float contracts (compiler.float-contracts)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod float_tests {
    use super::*;
    use z3::SatResult;

    fn lit_f(f: f64) -> ContractTerm {
        ContractTerm::LitFloat(ContractFloat(f))
    }
    fn lit(n: i64) -> ContractTerm {
        ContractTerm::LitInt(n)
    }
    fn cmp(op: CmpOp, lhs: ContractTerm, rhs: ContractTerm) -> ContractTerm {
        ContractTerm::Cmp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    #[test]
    fn lit_float_lt_works() {
        // `0.5 < 1.5` lowers to a Real comparison and asserts true.
        let term = cmp(CmpOp::Lt, lit_f(0.5), lit_f(1.5));
        let subst = Substitution::new();
        let out = lower_bool(&term, &subst).unwrap();
        let solver = z3::Solver::new();
        solver.assert(&out.term);
        assert_eq!(solver.check(), SatResult::Sat);
        // Conjugate with negation — should be unsat (the original is a
        // tautology under any model).
        let solver = z3::Solver::new();
        solver.assert(&out.term.not());
        assert_eq!(solver.check(), SatResult::Unsat);
    }

    #[test]
    fn mixed_int_float_cmp_coerces_to_real() {
        // `5 < 5.5`: LitInt vs LitFloat — coerces to Real comparison.
        let term = cmp(CmpOp::Lt, lit(5), lit_f(5.5));
        let subst = Substitution::new();
        let out = lower_bool(&term, &subst).unwrap();
        let solver = z3::Solver::new();
        solver.assert(&out.term.not());
        assert_eq!(solver.check(), SatResult::Unsat);
    }

    #[test]
    fn float_value_id_substitutes_real_term() {
        // Plant a Real Z3 term for ValueId(42); assert `Value(42) >= 0.0`
        // against the symbolic. The substitution provides the Real wire.
        let subst = Substitution::new().with_value_id_term(
            ValueId(42),
            Z3Term::Real(Real::fresh_const("v42_")),
        );
        let term = cmp(
            CmpOp::Ge,
            ContractTerm::Var(ContractVar::Value(ValueId(42))),
            lit_f(0.0),
        );
        // Lowering succeeds without sort-mismatch errors.
        let _ = lower_bool(&term, &subst).expect("float Var lowering should succeed");
    }

    #[test]
    fn float_arith_add_is_real_typed() {
        // `1.5 + 2.5 == 4.0` — arith propagates Real, comparison stays Real.
        let term = cmp(
            CmpOp::Eq,
            ContractTerm::Arith {
                op: ArithOp::Add,
                lhs: Box::new(lit_f(1.5)),
                rhs: Box::new(lit_f(2.5)),
            },
            lit_f(4.0),
        );
        let subst = Substitution::new();
        let out = lower_bool(&term, &subst).unwrap();
        let solver = z3::Solver::new();
        solver.assert(&out.term.not());
        assert_eq!(solver.check(), SatResult::Unsat);
    }
}
