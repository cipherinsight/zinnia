//! Structural-predicate machinery: registry, contracts, formula AST,
//! discharge orchestrator, and cache.
//!
//! ## Module layout
//!
//! - [`registry`] — predicate-kind table (e.g., `nnz` → Z3 encoder +
//!   meta-facts). Per-predicate cards (W1–W5) extend [`registry::build_registry`].
//! - [`contracts`] — `OpContract` + per-IR-kind registry. Per-predicate
//!   cards extend [`contracts::build_contract_registry`]. Currently
//!   ships empty contracts; the framework wiring is what's load-bearing
//!   here.
//! - [`formula`] — `ContractTerm` AST + Z3 lowering. The formula language
//!   for contract templates.
//! - [`discharge`] — the orchestrator: find structural-predicate atoms,
//!   bridge to Z3, accumulate clauses + meta-facts.
//! - [`cache`] — `DischargeCache` keyed on (target, obligation, slice).
//!
//! ## What this module exposes upstream
//!
//! `crate::optim::predicates` is the umbrella. Three external consumers:
//!
//! 1. [`smt_encode_structural_predicate`] — called by
//!    [`crate::optim::smt_encoding::IROp::smt_encode`] for the
//!    `IR::StructuralPredicate` arm. Encodes the atom + injects meta-facts
//!    into the `SmtEncodingCtx`.
//! 2. [`op_contract_for`] — called by the contracts framework / discharge
//!    layer when assembling per-op contracts at chokepoint time.
//! 3. [`Discharger`] — the orchestrator type. Cards downstream that wire
//!    contracts into the resolver hot path will instantiate one of these
//!    per compilation.
//!
//! ## Extension recipe (per-predicate cards)
//!
//! 1. Add a registry entry in [`registry::build_registry`].
//! 2. If the predicate has structural meaning for any IR op (e.g.,
//!    `nonzero` ensures `len(y) == nnz(x)`), add a contract entry in
//!    [`contracts::build_contract_registry`] using
//!    [`formula::ContractTerm`] builders.
//! 3. Add tests under [`tests`].

pub mod cache;
pub mod contracts;
pub mod discharge;
pub mod facts;
pub mod formula;
pub mod registry;
pub mod witness;

#[cfg(test)]
mod tests;

use z3::ast::Int;

use crate::ir_defs::IR;
use crate::optim::smt_encoding::{SmtEncodingCtx, Z3Term};

// ---------------------------------------------------------------------------
// Public re-exports
// ---------------------------------------------------------------------------

pub use cache::{DischargeCache, DischargeKey, DischargeResult};
pub use facts::{
    collect_value_ids, instantiate_contract, substitute_inputs, Fact, FactScope, FactSet,
    FactStack, ScopeKind,
};
pub use contracts::{op_contract_by_name, op_contract_for, ContractFormula, FrameCondition, OpContract};
pub use discharge::{
    build_input_array_lengths, build_input_name_index, find_scalar_preconditions,
    find_structural_predicates, Discharger, PredicateConstraints,
};
pub use formula::{
    lower_bool, ArithOp, BoolOp, CmpOp, ContractTerm, ContractVar, LowerError, LowerOutput,
    Substitution,
};
pub use registry::{registry, PredicateRegistration};
pub use witness::{witness_emitters, WitnessEmitter};

// ---------------------------------------------------------------------------
// SMT encoder for IR::StructuralPredicate
// ---------------------------------------------------------------------------

/// Encode an `IR::StructuralPredicate` atom as a Z3 `Bool` term, injecting
/// the predicate's meta-facts into `ctx` as a side effect.
///
/// Standalone in-place encoder used by the smt_encoding match-arm. For
/// SSA-bound encodings the discharger ([`Discharger`]) is the right home
/// — it can look up named-input Z3 terms via [`build_input_name_index`]
/// instead of minting fresh symbols.
///
/// Falls back to `ctx.fresh_unconstrained()` when the predicate kind is
/// unregistered or arity-mismatched. Both are sound (the atom contributes
/// no information rather than wrong information).
pub fn smt_encode_structural_predicate(ir: &IR, ctx: &mut SmtEncodingCtx) -> Z3Term {
    if let IR::StructuralPredicate { kind, args, op, bound } = ir {
        if let Some(term) = encode_atom_fresh(ctx, kind, args, op.as_deref(), bound.as_deref()) {
            return term;
        }
    }
    ctx.fresh_unconstrained()
}

/// Inner encoder: mint fresh Z3 Ints per arg, build the predicate term,
/// inject meta-facts. Returns `None` if the kind is unregistered or the
/// arity is wrong.
fn encode_atom_fresh(
    ctx: &mut SmtEncodingCtx,
    kind: &str,
    args: &[String],
    op: Option<&str>,
    bound: Option<&str>,
) -> Option<Z3Term> {
    use z3::ast::Bool;

    let reg = registry().get(kind)?;
    if args.len() != reg.arity {
        return None;
    }

    // Fresh symbolic Int per predicate arg. Naming includes the source-
    // level identifier so SMT-LIB dumps stay readable.
    let arg_ints: Vec<Int> = args
        .iter()
        .map(|name| Int::fresh_const(&format!("sp_arg_{name}_")))
        .collect();

    // Inject meta-facts (deduped per-query).
    let facts = reg.meta_facts(&arg_ints);
    ctx.inject_meta_facts(kind, facts);

    // Predicate application.
    let app = reg.encode_app(&arg_ints);

    // Optional comparison-with-bound clause.
    let result = match (op, bound) {
        (Some(op_str), Some(bound_name)) => {
            let bound_int = Int::fresh_const(&format!("sp_bound_{bound_name}_"));
            let pred_value = Int::fresh_const(&format!("sp_pred_value_{kind}_"));
            let relation = match compare_z3(&pred_value, op_str, &bound_int) {
                Some(r) => r,
                None => return None,
            };
            Bool::and(&[&app, &relation])
        }
        _ => app,
    };

    Some(Z3Term::Bool(result))
}

fn compare_z3(lhs: &Int, op: &str, rhs: &Int) -> Option<z3::ast::Bool> {
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
