//! Exact field negation of a difference: -(a - b) = b - a.
//! Only unrounded SubI/SubF and raw integer multiplication are rewritten;
//! rounded MulF, AbsF and checked negations retain their domain checks.
use super::IRPass;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::StmtId;

pub struct NeutralNegation;

fn constant(graph: &IRGraph, id: StmtId, int: i64) -> bool {
    constant_with_budget(graph, id, int, 8)
}

fn constant_with_budget(graph: &IRGraph, id: StmtId, int: i64, budget: usize) -> bool {
    if budget == 0 {
        return false;
    }
    let stmt = &graph.stmts[id as usize];
    match stmt.ir {
        IR::ConstantInt { value } => value == int,
        IR::ConstantFloat { value } => int == 0 && value == 0.0,
        // Complex multiplication leaves raw `0 * component` nodes. These
        // are field products without a rescale or operand-domain check.
        IR::MulI if int == 0 && stmt.arguments.len() == 2 => stmt
            .arguments
            .iter()
            .any(|&a| constant_with_budget(graph, a, 0, budget - 1)),
        _ => false,
    }
}

impl IRPass for NeutralNegation {
    fn exec(&self, mut graph: IRGraph) -> IRGraph {
        // Reversing a shared difference materializes both signs. Backends
        // can consume its negation as a coefficient instead, so only
        // reverse a difference whose original materialization disappears.
        // Uses in raw zero products do not require that materialization.
        let mut uses = vec![0usize; graph.stmts.len()];
        for stmt in &graph.stmts {
            if matches!(stmt.ir, IR::MulI) && constant(&graph, stmt.stmt_id, 0) {
                continue;
            }
            for &a in &stmt.arguments {
                uses[a as usize] += 1;
            }
        }
        let mut stmts = graph.stmts.clone();
        for stmt in &mut stmts {
            let inner = match (&stmt.ir, stmt.arguments.as_slice()) {
                (IR::SubI | IR::SubF, &[zero, x]) if constant(&graph, zero, 0) => Some(x),
                (IR::MulI, &[a, x]) if constant(&graph, a, -1) => Some(x),
                (IR::MulI, &[x, b]) if constant(&graph, b, -1) => Some(x),
                _ => None,
            };
            let Some(inner) = inner.map(|id| &graph.stmts[id as usize]) else {
                continue;
            };
            if matches!(inner.ir, IR::SubI | IR::SubF)
                && inner.arguments.len() == 2
                && uses[inner.stmt_id as usize] == 1
            {
                stmt.ir = inner.ir.clone();
                stmt.arguments = vec![inner.arguments[1], inner.arguments[0]];
                stmt.arg_values.clear();
            }
        }
        graph.update_graph(stmts);
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IRStatement;
    use crate::types::ValueId;
    #[test]
    fn neutral_negation_reverses_exact_difference_but_keeps_rounded_product() {
        let nodes = vec![
            (
                IR::ReadFloat {
                    path: crate::circuit_input::InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            (
                IR::ReadFloat {
                    path: crate::circuit_input::InputPath::new("y", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            (IR::ConstantInt { value: -1 }, vec![]),
            (IR::ConstantFloat { value: 0.0 }, vec![]),
            (IR::SubF, vec![0, 1]),
            (IR::MulI, vec![2, 4]),
            (IR::SubF, vec![0, 1]),
            (IR::SubF, vec![3, 6]),
            (IR::ConstantFloat { value: -1.0 }, vec![]),
            (IR::SubF, vec![0, 1]),
            (IR::MulF, vec![8, 9]),
            (IR::ConstantInt { value: 0 }, vec![]),
            (IR::MulI, vec![11, 0]),
            (IR::SubF, vec![0, 1]),
            (IR::SubF, vec![12, 13]),
            (IR::MulI, vec![11, 13]), // zero use must not block reversal
            (IR::SubF, vec![0, 1]),
            (IR::SubF, vec![3, 16]),
            (IR::ExposePublicF, vec![16]), // genuinely shared difference
        ];
        let graph = IRGraph::new(
            nodes
                .into_iter()
                .enumerate()
                .map(|(id, (ir, args))| {
                    IRStatement::new(id as StmtId, ValueId::next(), ir, args, vec![], None)
                })
                .collect(),
        );
        let out = NeutralNegation.exec(graph);
        for id in [5, 7, 14] {
            assert!(matches!(out.stmts[id].ir, IR::SubF));
            assert_eq!(out.stmts[id].arguments, vec![1, 0]);
        }
        assert!(matches!(out.stmts[10].ir, IR::MulF));
        assert_eq!(out.stmts[10].arguments, vec![8, 9]);
        assert_eq!(
            out.stmts[17].arguments,
            vec![3, 16],
            "keep a shared difference available with coefficient -1"
        );
    }
}
