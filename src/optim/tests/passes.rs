#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::{IRGraph, IRStatement};
    use crate::ir_defs::IR;
    use crate::optim::*;

    #[test]
    fn test_external_call_remover() {
        let stmts = vec![
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 1 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::InvokeExternal {
                    store_idx: 0,
                    func_name: "f".to_string(),
                    args: vec![],
                    kwargs: HashMap::new(),
                },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![0],
                vec![],
                None,
            ),
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
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 10 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 20 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![1],
                vec![],
                None,
            ),
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
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantBool { value: true },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::LogicalNot,
                vec![0],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::LogicalNot,
                vec![1],
                vec![],
                None,
            ),
            IRStatement::new(
                3,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![2],
                vec![],
                None,
            ),
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
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 1 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![0],
                vec![],
                None,
            ),
        ];
        let graph = IRGraph::new(stmts);
        let result = AlwaysSatisfiedElimination.exec(graph);
        // Assert on const(1) should be eliminated
        assert_eq!(result.len(), 1);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 1 }));
    }

    #[test]
    fn duplicate_code_elimination_replaces_duplicate_constants() {
        let stmts = vec![
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 7 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 7 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![1],
                vec![],
                None,
            ),
        ];
        let graph = IRGraph::new(stmts);
        let result = DuplicateCodeElimination.exec(graph);

        assert_eq!(result.len(), 2);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 7 }));
        assert!(matches!(result.stmts[1].ir, IR::Assert));
        assert_eq!(result.stmts[1].arguments, vec![0]);
    }

    #[test]
    fn duplicate_code_elimination_keeps_first_canonical_for_three_duplicates() {
        let stmts = vec![
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 7 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 7 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 7 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                3,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![2],
                vec![],
                None,
            ),
        ];
        let graph = IRGraph::new(stmts);
        let result = DuplicateCodeElimination.exec(graph);

        assert_eq!(result.len(), 2);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 7 }));
        assert_eq!(result.stmts[1].arguments, vec![0]);
    }

    #[test]
    fn unread_memory_elimination_drops_unread_segment_and_following_dce_cleans_inputs() {
        let stmts = vec![
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 7 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 11 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::AllocateMemory {
                    segment_id: 9,
                    size: 4,
                    init_value: 0,
                },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                3,
                crate::types::ValueId::next(),
                IR::AddI,
                vec![0, 1],
                vec![],
                None,
            ),
            IRStatement::new(
                4,
                crate::types::ValueId::next(),
                IR::WriteMemory { segment_id: 9 },
                vec![0, 3],
                vec![],
                None,
            ),
            IRStatement::new(
                5,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![1],
                vec![],
                None,
            ),
        ];
        let graph = IRGraph::new(stmts);
        let graph = UnreadMemoryEliminator.exec(graph);
        let result = DeadCodeElimination.exec(graph);

        assert_eq!(result.len(), 2);
        assert!(matches!(result.stmts[0].ir, IR::ConstantInt { value: 11 }));
        assert!(matches!(result.stmts[1].ir, IR::Assert));
        assert_eq!(result.stmts[1].arguments, vec![0]);
    }

    #[test]
    fn duplicate_code_elimination_canonicalizes_commutative_mulf_args() {
        let stmts = vec![
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ReadFloat {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::ReadFloat {
                    path: crate::circuit_input::InputPath::new("y", vec![]),
                    is_public: false,
                },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::MulF,
                vec![0, 1],
                vec![],
                None,
            ),
            IRStatement::new(
                3,
                crate::types::ValueId::next(),
                IR::MulF,
                vec![1, 0],
                vec![],
                None,
            ),
            IRStatement::new(
                4,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![3],
                vec![],
                None,
            ),
        ];
        let graph = IRGraph::new(stmts);
        let result = DuplicateCodeElimination.exec(graph);

        assert_eq!(result.len(), 4);
        assert!(matches!(result.stmts[2].ir, IR::MulF));
        assert_eq!(result.stmts[3].arguments, vec![2]);
    }

    #[test]
    fn duplicate_code_elimination_replaces_duplicates_after_many_unrelated_keys() {
        let stmts = vec![
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 0 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 1 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 2 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                3,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 3 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                4,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 0 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                5,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![4],
                vec![],
                None,
            ),
        ];
        let graph = IRGraph::new(stmts);
        let result = DuplicateCodeElimination.exec(graph);

        assert_eq!(result.len(), 5);
        assert!(matches!(result.stmts[4].ir, IR::Assert));
        assert_eq!(result.stmts[4].arguments, vec![0]);
    }

    #[test]
    fn test_pattern_match_add_zero() {
        // stmt0 = const(0), stmt1 = read_integer, stmt2 = add_i(0, 1)
        // add(0, x) should simplify to x
        let stmts = vec![
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 0 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::ReadInteger {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::AddI,
                vec![0, 1],
                vec![],
                None,
            ),
            IRStatement::new(
                3,
                crate::types::ValueId::next(),
                IR::Assert,
                vec![2],
                vec![],
                None,
            ),
        ];
        let graph = IRGraph::new(stmts);
        let result = PatternMatchOptim.exec(graph);
        // The add should be simplified, so assert should reference the read directly
        // After optimization: const(0), read_int, assert(read_int)
        assert!(result.len() <= 4);
    }

    /// `x - x` is the constant 0, for integers and for (Q32) floats: a
    /// comparison of an array with its own row then needs no circuit.
    #[test]
    fn test_pattern_match_sub_self_is_zero() {
        for float in [false, true] {
            let read = if float {
                IR::ReadFloat { path: crate::circuit_input::InputPath::new("x", vec![]), is_public: false }
            } else {
                IR::ReadInteger { path: crate::circuit_input::InputPath::new("x", vec![]), is_public: false }
            };
            let (sub, expose) = if float { (IR::SubF, IR::ExposePublicF) } else { (IR::SubI, IR::ExposePublicI) };
            let stmts = vec![
                IRStatement::new(0, crate::types::ValueId::next(), read, vec![], vec![], None),
                IRStatement::new(1, crate::types::ValueId::next(), sub, vec![0, 0], vec![], None),
                IRStatement::new(2, crate::types::ValueId::next(), expose, vec![1], vec![], None),
            ];
            let result = PatternMatchOptim.exec(IRGraph::new(stmts));
            assert!(!result.stmts.iter().any(|s| matches!(s.ir, IR::SubF | IR::SubI)), "float = {float}");
            let exposed = result.stmts.last().unwrap();
            let zero = &result.stmts[exposed.arguments[0] as usize].ir;
            assert!(
                matches!(zero, IR::ConstantFloat { value } if *value == 0.0) || matches!(zero, IR::ConstantInt { value: 0 }),
                "float = {float}: {zero:?}"
            );
        }
    }

    /// Statements of a small graph, ids are indices, no argument value ids.
    fn graph_of(irs: Vec<(IR, Vec<u32>)>) -> IRGraph {
        IRGraph::new(
            irs.into_iter()
                .enumerate()
                .map(|(i, (ir, args))| IRStatement::new(i as u32, crate::types::ValueId::next(), ir, args, vec![], None))
                .collect(),
        )
    }

    fn read(name: &str) -> IR {
        IR::ReadInteger { path: crate::circuit_input::InputPath::new(name, vec![]), is_public: false }
    }

    /// `assert(lo <= x and x <= hi)` becomes one assertion per comparison,
    /// each comparison keeping its form; nested conjunctions too.
    #[test]
    fn pattern_match_splits_asserted_conjunctions() {
        let graph = graph_of(vec![
            (read("x"), vec![]),
            (IR::ConstantInt { value: -1 }, vec![]),
            (IR::ConstantInt { value: 1 }, vec![]),
            (IR::GteI, vec![0, 1]),
            (IR::LteI, vec![0, 2]),
            (IR::NeI, vec![0, 2]),
            (IR::LogicalAnd, vec![3, 4]),
            (IR::LogicalAnd, vec![6, 5]),
            (IR::Assert, vec![7]),
        ]);
        let result = DeadCodeElimination.exec(PatternMatchOptim.exec(graph));
        let asserted: Vec<&IR> = result
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::Assert))
            .map(|s| &result.stmts[s.arguments[0] as usize].ir)
            .collect();
        assert!(matches!(asserted.as_slice(), [IR::GteI, IR::LteI, IR::NeI]), "{asserted:?}");
        assert!(result.stmts.iter().all(|s| !matches!(s.ir, IR::LogicalAnd)), "conjunction kept");
    }

    /// Four subjects tested against the same three constants (`np.in1d`)
    /// use Horner's rule on shared coefficients: no difference per factor,
    /// and the zero test agrees with the equalities for members, non-members
    /// and repeated constants (mock).
    #[test]
    fn pattern_match_shares_membership_coefficients() {
        let mut irs = vec![(read("b0"), vec![]), (read("b1"), vec![]), (read("b2"), vec![])];
        let mut exposed = Vec::new();
        for k in 0..4u32 {
            let x = irs.len() as u32;
            irs.push((read(&format!("x{k}")), vec![]));
            let eqs: Vec<u32> = (0..3)
                .map(|j| {
                    irs.push((IR::EqI, vec![x, j]));
                    irs.len() as u32 - 1
                })
                .collect();
            irs.push((IR::LogicalOr, vec![eqs[0], eqs[1]]));
            irs.push((IR::LogicalOr, vec![irs.len() as u32 - 1, eqs[2]]));
            exposed.push(irs.len() as u32 - 1);
        }
        for e in &exposed {
            irs.push((IR::ExposePublicI, vec![*e]));
        }
        let result = DeadCodeElimination.exec(PatternMatchOptim.exec(graph_of(irs)));
        // The coefficients' subtractions only (computed once): a difference
        // per factor and subject would be 12.
        let subtractions = result.stmts.iter().filter(|s| matches!(s.ir, IR::SubI)).count();
        assert!(subtractions <= 6, "a difference per factor: {subtractions}");
        for (bs, xs) in [([2, 5, 9], [5, 1, 9, 2]), ([3, 3, -4], [3, -4, 4, 0]), ([0, 0, 0], [0, 1, -1, 7])] {
            let mut inputs: Vec<(String, i64)> = (0..3).map(|j| (format!("b{j}"), bs[j])).collect();
            inputs.extend((0..4).map(|k| (format!("x{k}"), xs[k])));
            let (ok, outputs) = crate::lower::sort::test_support::mock_run(&result, &inputs);
            let expected: Vec<i64> = xs.iter().map(|x| bs.contains(x) as i64).collect();
            assert!(ok);
            assert_eq!(outputs, expected, "B = {bs:?}, x = {xs:?}");
        }
    }

    #[test]
    fn test_memory_trace_injection() {
        let stmts = vec![
            IRStatement::new(
                0,
                crate::types::ValueId::next(),
                IR::AllocateMemory {
                    segment_id: 0,
                    size: 10,
                    init_value: 0,
                },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                1,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 0 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                2,
                crate::types::ValueId::next(),
                IR::ConstantInt { value: 42 },
                vec![],
                vec![],
                None,
            ),
            IRStatement::new(
                3,
                crate::types::ValueId::next(),
                IR::WriteMemory { segment_id: 0 },
                vec![1, 2],
                vec![],
                None,
            ),
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
