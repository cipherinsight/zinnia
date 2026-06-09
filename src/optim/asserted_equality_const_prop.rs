//! Asserted-equality constant propagation pass.
//!
//! When the program asserts `x == const`, every downstream consumer of `x`
//! could safely treat `x` as `const`. nargo's compiler does this naturally
//! at the Noir level (which is why a `assert(size[0] == 3); let r = size[0]
//! / 2;` block folds `r` to `1` even when `size[0]` is a runtime witness).
//! Zinnia's IR-build constant folder consults `int_val()` only — it doesn't
//! look at facts — so the same downstream computation stays runtime in the
//! Zinnia emit even though the fact is already deposited.
//!
//! This pass closes that gap as a post-IR-build rewrite:
//!
//! 1. Walk the IR; for every `IR::Assert` whose argument resolves to the
//!    AST→IR `assert_value` shape (`SelectI(ConstantBool(true), EqI(x, k),
//!    ConstantBool(true))`), record `(stmt_id(x), k)`.
//!    The `SelectI` wrapper carries the path condition; we only fold the
//!    unconditional case — conditional asserts could be inside an `if` that
//!    the prover-side never enters, so propagating their constant downstream
//!    would be unsound.
//! 2. Rebuild the IR. For each statement, replace any arg whose `stmt_id`
//!    appears in the recorded map with a fresh `IR::ConstantInt { value: k }`.
//!    Skip stmts that are *part of the assert chain itself* (the `EqI`, the
//!    `SelectI`, and the `Assert`) — propagating into those would simplify
//!    them to trivially-true and the `AlwaysSatisfiedElimination` pass would
//!    then remove the runtime check, which is the only thing tying the
//!    witness to the asserted value at prove time. The propagation is sound
//!    only if the runtime assert survives.
//!
//! The constant_fold pass that runs afterward picks up the new constant
//! leaves and folds the downstream arithmetic the rest of the way.

use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct AssertedEqualityConstantPropagation;

impl IRPass for AssertedEqualityConstantPropagation {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let stmts = &ir_graph.stmts;

        // Phase 1: collect (var_stmt_id -> constant) from unconditional asserts.
        let mut sub: HashMap<StmtId, i64> = HashMap::new();
        // Stmts that belong to assert chains; we must NOT substitute into them
        // (otherwise the runtime check becomes trivially-true and DCE-equivalent
        // passes will drop it, leaving the witness unconstrained).
        let mut keep_intact: HashSet<StmtId> = HashSet::new();

        for stmt in stmts {
            if !matches!(stmt.ir, IR::Assert) {
                continue;
            }
            if stmt.arguments.len() != 1 {
                continue;
            }
            let select_id = stmt.arguments[0];
            let select_stmt = match stmts.get(select_id as usize) {
                Some(s) => s,
                None => continue,
            };
            // assert_value wraps the user condition as:
            //   conditioned = SelectI(path_cond, scalar, true)
            // We only fold the unconditional case where path_cond is
            // statically `true`.
            if !matches!(select_stmt.ir, IR::SelectI) || select_stmt.arguments.len() != 3 {
                continue;
            }
            let cond_id = select_stmt.arguments[0];
            let cond_stmt = match stmts.get(cond_id as usize) {
                Some(s) => s,
                None => continue,
            };
            if !matches!(cond_stmt.ir, IR::ConstantBool { value: true }) {
                continue;
            }
            let eq_id = select_stmt.arguments[1];
            let eq_stmt = match stmts.get(eq_id as usize) {
                Some(s) => s,
                None => continue,
            };
            if !matches!(eq_stmt.ir, IR::EqI) || eq_stmt.arguments.len() != 2 {
                continue;
            }
            let lhs_id = eq_stmt.arguments[0];
            let rhs_id = eq_stmt.arguments[1];
            let lhs_stmt = match stmts.get(lhs_id as usize) {
                Some(s) => s,
                None => continue,
            };
            let rhs_stmt = match stmts.get(rhs_id as usize) {
                Some(s) => s,
                None => continue,
            };
            let (var_id, k) = match (&lhs_stmt.ir, &rhs_stmt.ir) {
                (IR::ConstantInt { value }, _) => (rhs_id, *value),
                (_, IR::ConstantInt { value }) => (lhs_id, *value),
                _ => continue,
            };

            // Don't fold-substitute into a freshly-emitted ConstantInt
            // (no-op), and don't shadow an already-fast-path constant.
            if let Some(s) = stmts.get(var_id as usize) {
                if matches!(s.ir, IR::ConstantInt { .. }) {
                    continue;
                }
            }

            sub.insert(var_id, k);
            keep_intact.insert(eq_id);
            keep_intact.insert(select_id);
            keep_intact.insert(stmt.stmt_id);
        }

        if sub.is_empty() {
            return ir_graph;
        }

        // Phase 2: rebuild with substitutions applied outside the assert chains.
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut const_cache: HashMap<i64, Value> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let apply_sub = !keep_intact.contains(&stmt.stmt_id);
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| {
                    if apply_sub {
                        if let Some(&k) = sub.get(&arg) {
                            return const_cache
                                .entry(k)
                                .or_insert_with(|| builder.ir_constant_int(k))
                                .clone();
                        }
                    }
                    value_lookup[&arg].clone()
                })
                .collect();
            let new_val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, new_val);
        }

        let mut new_graph = builder.export_ir_graph();
        new_graph.resolver_mut().on_ir_mutated(&[]);
        new_graph
    }
}
