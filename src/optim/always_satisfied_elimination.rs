use std::collections::HashMap;
use std::sync::atomic::Ordering;

use crate::builder::IRBuilder;
use crate::error::ZinniaError;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{ScalarValue, StmtId, Value};

use super::IRPass;

/// Result of the elimination pass — distinguishes the OK path from a
/// compile-time failure (a provably-false assertion). `IRPass::exec` today
/// returns an `IRGraph`, so the pass keeps panicking on the failure case;
/// switching to `Result<IRGraph, ZinniaError>` is a wider IRPass-trait
/// change which we leave for a future refactor (the spec calls out the
/// design surface but doesn't mandate the wider change for round 1).
pub struct AlwaysSatisfiedElimination;

impl IRPass for AlwaysSatisfiedElimination {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
        // Phase 1: re-build constants into a side `IRBuilder` so we have
        // each stmt's static_val (the pre-P4 path). This keeps the
        // cheap constant-fold elimination working without any resolver
        // call — most assertions in practice are folded by constant
        // propagation alone.
        let mut builder = IRBuilder::new();
        let mut values_lookup: HashMap<StmtId, Value> = HashMap::new();
        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| values_lookup[&arg].clone())
                .collect();
            let val = builder.create_ir(&stmt.ir, &ir_args);
            values_lookup.insert(stmt.stmt_id, val);
        }

        // Snapshot the resolver's telemetry handle (if any) before we
        // borrow `ir_graph` mutably for the elimination walk. The handle
        // is `Arc`, so this is cheap — and it lets us bump the
        // `assertions_eliminated_*` counters without a second
        // `&mut ir_graph` borrow.
        let telemetry = ir_graph.resolver_telemetry();

        // Phase 2: pick out the assertions to eliminate (or reject).
        // Cheap path: the side-build's static_val. Expensive fallback:
        // the active resolver, walking the live IR statements via the
        // `_with_stmts` chokepoint.
        let mut to_eliminate: Vec<StmtId> = Vec::new();
        let mut provably_false_asserts: Vec<(StmtId, StmtId)> = Vec::new();

        // Collect the assert candidates first so we don't hold an immutable
        // borrow on `ir_graph.stmts` across the resolver call (the resolver
        // wants its own `(&mut Resolver, &[IRStatement])` split-borrow).
        let assert_candidates: Vec<(StmtId, StmtId)> = ir_graph
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::Assert))
            .map(|s| (s.stmt_id, s.arguments[0]))
            .collect();

        for (assert_stmt_id, cond_ptr) in assert_candidates {
            // Cheap path — try the side-built static_val first.
            if let Some(cond_val) = values_lookup.get(&cond_ptr) {
                if let Some(v) = cond_val.int_val() {
                    if v != 0 {
                        to_eliminate.push(assert_stmt_id);
                        if let Some(t) = telemetry.as_ref() {
                            t.assertions_eliminated_const_fold
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        continue;
                    }
                    // v == 0 — provably false on the constant-fold path.
                    provably_false_asserts.push((assert_stmt_id, cond_ptr));
                    if let Some(t) = telemetry.as_ref() {
                        t.assertions_provably_false
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    continue;
                }
            }

            // Expensive path — the resolver gets a shot. Build a
            // `Value::Boolean` with the cond's stmt id as ptr; the
            // `_with_stmts` impl walks the IR from that pointer.
            //
            // Path-condition awareness: the AST→IR phase emits assertions
            // as `assert(select(path_cond, original_cond, true))`
            // (see `IRGenerator::assert_value` in `src/ir_gen/visitors.rs`),
            // so the cond passed to the resolver already includes the
            // surrounding control flow's path condition. No extra
            // Resolver API parameter is needed.
            let probe = Value::Boolean(ScalarValue::runtime(cond_ptr));
            let (resolver, stmts) = ir_graph.split_resolver_and_stmts();
            let resolved = resolver.resolve_bool_with_stmts(&probe, stmts);
            match resolved {
                Some(true) => {
                    to_eliminate.push(assert_stmt_id);
                    if let Some(t) = telemetry.as_ref() {
                        t.assertions_eliminated_resolver
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                Some(false) => {
                    provably_false_asserts.push((assert_stmt_id, cond_ptr));
                    if let Some(t) = telemetry.as_ref() {
                        t.assertions_provably_false
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => {
                    // Resolver couldn't prove anything — leave the assert
                    // in place. Today's pre-P4 fallback for non-constant
                    // assertions.
                }
            }
        }

        // Phase 3: a provably-false assertion is a hard compile-time
        // error. Mirrors the pre-Rust Python pass at `op_assert.py:38`
        // which raised `StaticInferenceError`. No matching Rust error
        // type exists yet — `ZinniaError` (the public bridge error) is
        // the closest equivalent. Emitted via panic to flow through the
        // pyo3 PyErr conversion the same way today's optim-time errors
        // do (`builder.rs:848`, `helpers/value_ops.rs:489`, etc.).
        if let Some((assert_stmt_id, cond_ptr)) = provably_false_asserts.first() {
            let err = ZinniaError {
                message: format!(
                    "static inference: assertion at stmt {} (cond stmt {}) is provably unsatisfiable",
                    assert_stmt_id, cond_ptr,
                ),
            };
            // Today's optim-time panics surface as PyRuntimeErrors via
            // pyo3; same exit channel here.
            panic!("{}", err);
        }

        ir_graph.remove_stmt_bunch(&to_eliminate);
        // P0 seam: signal mutation to the resolver. No-op today via
        // StaticOnlyResolver; P1 uses this for cache invalidation.
        ir_graph.resolver_mut().on_ir_mutated(&[]);
        ir_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IRGraph, IRStatement};
    use crate::ir_defs::IR;
    use crate::optim::resolver::LayeredResolver;

    /// Pre-P4 path — constant-fold catches `assert(1)` and removes it.
    #[test]
    fn assert_eliminated_when_constant_fold_proves_true() {
        let stmts = vec![
            IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantInt { value: 1 }, vec![], vec![], None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::Assert, vec![0], vec![], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = AlwaysSatisfiedElimination.exec(graph);
        assert_eq!(result.len(), 1);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 1 }));
    }

    /// P4 round 1 — the resolver path catches `assert(x == x)` for a
    /// free `x`. Constant-fold cannot fold this (both sides are runtime
    /// reads), but the SMT layer proves `x == x` is `Some(true)`
    /// universally, and the assertion gets removed.
    #[test]
    fn assert_eliminated_when_resolver_proves_true() {
        // stmt0 = read_int x, stmt1 = eq(x, x), stmt2 = assert(stmt1)
        let stmts = vec![
            IRStatement::new(
                0, crate::types::ValueId::next(),
                IR::ReadInteger {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![], vec![],
                None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::EqI, vec![0, 0], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::Assert, vec![1], vec![], None),
        ];
        let mut graph = IRGraph::new(stmts);
        // Wire the layered resolver — same composition the real default
        // pipeline uses (range → SMT). Without this the side-build's
        // static_val fast-path hands back None for `EqI` of two reads,
        // and the optim pass leaves the assert in place.
        graph.set_resolver(Box::new(LayeredResolver::range_then_smt()));

        let result = AlwaysSatisfiedElimination.exec(graph);
        // The assert should have been eliminated.
        let remaining_asserts = result
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::Assert))
            .count();
        assert_eq!(
            remaining_asserts, 0,
            "expected resolver-proved assert(x == x) to be eliminated"
        );
    }

    /// P4 round 1 — `assert(false)` (or constant 0) raises a compile-
    /// time error rather than silently lingering as a runtime check.
    #[test]
    #[should_panic(expected = "provably unsatisfiable")]
    fn assert_rejected_when_constant_fold_proves_false() {
        let stmts = vec![
            IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: false }, vec![], vec![], None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::Assert, vec![0], vec![], None),
        ];
        let graph = IRGraph::new(stmts);
        let _ = AlwaysSatisfiedElimination.exec(graph);
    }
}
