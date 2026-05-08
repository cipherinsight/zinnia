// Note: This file is included as `#[cfg(test)] mod tests;` from mod.rs,
// so it's already test-only. No inner `mod tests` wrapper needed.

use super::*;

#[test]
fn test_generate_simple_constants() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                    "value": {"__class__": "ASTConstantInteger", "value": 42}
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        assert!(!graph.is_empty()); // At least the constant + type registrations
    }

    #[test]
    fn test_generate_binary_op() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "z"}],
                    "value": {
                        "__class__": "ASTBinaryOperator",
                        "operator": "add",
                        "lhs": {"__class__": "ASTConstantInteger", "value": 10},
                        "rhs": {"__class__": "ASTConstantInteger", "value": 20}
                    }
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        // Should have: constant(10), constant(20), add_i
        assert!(graph.len() >= 3);
    }

    #[test]
    fn test_generate_if_else() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "cond"}],
                    "value": {"__class__": "ASTConstantBoolean", "value": true}
                },
                {
                    "__class__": "ASTCondStatement",
                    "cond": {"__class__": "ASTLoad", "name": "cond"},
                    "t_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 1}
                        }
                    ],
                    "f_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 2}
                        }
                    ]
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        assert!(graph.len() >= 3);
    }

    /// P4 round 1 — while-loop early termination via the resolver.
    ///
    /// A `while i < 5: i += 1` loop with starting `i = 0` must terminate
    /// after 5 unrolled iterations, well before `loop_limit` (default 256).
    /// Today this works through the static-val fast-path inside
    /// `bool_val()` / `int_val()`, but the test pins the behaviour: any
    /// regression in the resolver routing in `visit_while` (e.g. wrong
    /// pre-/post-iteration env capture) would make the loop unroll to its
    /// full 256-iteration cap and emit thousands of guarded statements.
    ///
    /// We pick a tight `loop_limit = 32`: if the early-exit fires after
    /// 5 iterations the resulting graph is small (~30 stmts); if it
    /// regresses to "always unroll", the graph balloons to 32+
    /// guarded body emissions.
    #[test]
    fn test_while_static_guard_early_termination() {
        // while i < 5: i = i + 1   (start i = 0)
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "i"}],
                    "value": {"__class__": "ASTConstantInteger", "value": 0}
                },
                {
                    "__class__": "ASTWhileStatement",
                    "test_expr": {
                        "__class__": "ASTBinaryOperator",
                        "operator": "lt",
                        "lhs": {"__class__": "ASTLoad", "name": "i"},
                        "rhs": {"__class__": "ASTConstantInteger", "value": 5}
                    },
                    "block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "i"}],
                            "value": {
                                "__class__": "ASTBinaryOperator",
                                "operator": "add",
                                "lhs": {"__class__": "ASTLoad", "name": "i"},
                                "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                            }
                        }
                    ]
                }
            ],
            "inputs": []
        }"#;

        let mut cfg = IRGenConfig::default();
        cfg.loop_limit = 32;
        let graph = IRGenerator::generate_from_json(cfg, json).unwrap();

        // The early-exit path should have fired at iteration 5
        // (when `i` first reaches 5 and the static guard becomes false).
        // Each iteration produces ~5 IR statements (load i, const 5, lt,
        // load i, const 1, add). 5 iterations × ~6 stmts ≈ 30 stmts.
        // A regression to full unrolling would yield 32 iterations of
        // guarded bodies — ≥ 100+ stmts.
        assert!(
            graph.len() < 80,
            "expected early termination, got {} stmts (regression?)",
            graph.len()
        );
    }
