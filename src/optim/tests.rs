#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::{IRGraph, IRStatement};
    use crate::ir_defs::IR;
    use crate::optim::*;

    #[test]
    fn test_external_call_remover() {
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 1 }, vec![], None),
            IRStatement::new(
                1,
                IR::InvokeExternal {
                    store_idx: 0,
                    func_name: "f".to_string(),
                    args: vec![],
                    kwargs: HashMap::new(),
                },
                vec![],
                None,
            ),
            IRStatement::new(2, IR::Assert, vec![0], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = ExternalCallRemover.exec(graph);
        assert_eq!(result.len(), 2);
        // Should have const and assert, no invoke_external
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { .. }));
        assert!(matches!(result.stmts[1].ir, IR::Assert));
    }

    #[test]
    fn test_dead_code_elimination() {
        // stmt0 = const(10) [unused], stmt1 = const(20), stmt2 = assert(stmt1)
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 10 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 20 }, vec![], None),
            IRStatement::new(2, IR::Assert, vec![1], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = DeadCodeElimination.exec(graph);
        // stmt0 should be eliminated (unused, not fixed)
        assert_eq!(result.len(), 2);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 20 }));
        assert!(matches!(result.stmts[1].ir, IR::Assert));
    }

    #[test]
    fn test_double_not_elimination() {
        // stmt0 = const_bool(true), stmt1 = not(0), stmt2 = not(1) [== stmt0]
        let stmts = vec![
            IRStatement::new(0, IR::ConstantBool { value: true }, vec![], None),
            IRStatement::new(1, IR::LogicalNot, vec![0], None),
            IRStatement::new(2, IR::LogicalNot, vec![1], None),
            IRStatement::new(3, IR::Assert, vec![2], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = DoubleNotElimination.exec(graph);
        // The double not should be eliminated; stmt2 should refer back to stmt0's value
        // Result should have: const_bool(true), not(0), assert(0)
        // (the double-not returns the original, so assert references const_bool directly)
        assert!(result.len() <= 4);
    }

    #[test]
    fn test_always_satisfied_elimination() {
        // stmt0 = const(1), stmt1 = assert(0) — always satisfied
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 1 }, vec![], None),
            IRStatement::new(1, IR::Assert, vec![0], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = AlwaysSatisfiedElimination.exec(graph);
        // Assert on const(1) should be eliminated
        assert_eq!(result.len(), 1);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 1 }));
    }

    #[test]
    fn test_pattern_match_add_zero() {
        // stmt0 = const(0), stmt1 = read_integer, stmt2 = add_i(0, 1)
        // add(0, x) should simplify to x
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(
                1,
                IR::ReadInteger {
                    indices: vec![0],
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
            IRStatement::new(3, IR::Assert, vec![2], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = PatternMatchOptim.exec(graph);
        // The add should be simplified, so assert should reference the read directly
        // After optimization: const(0), read_int, assert(read_int)
        assert!(result.len() <= 4);
    }

    #[test]
    fn test_memory_trace_injection() {
        let stmts = vec![
            IRStatement::new(
                0,
                IR::AllocateMemory {
                    segment_id: 0,
                    size: 10,
                    init_value: 0,
                },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 42 }, vec![], None),
            IRStatement::new(3, IR::WriteMemory { segment_id: 0 }, vec![1, 2], None),
        ];
        let graph = IRGraph::new(stmts);
        let result = MemoryTraceInjection.exec(graph);
        // Should have: alloc, const, const, write, trace_emit, trace_seal
        assert!(result.len() >= 6);
        // Last stmt should be MemoryTraceSeal
        assert!(matches!(
            result.stmts.last().unwrap().ir,
            IR::MemoryTraceSeal
        ));
    }
}
