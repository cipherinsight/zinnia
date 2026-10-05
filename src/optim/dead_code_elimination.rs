use crate::ir::IRGraph;
use crate::types::StmtId;

use super::IRPass;

pub struct DeadCodeElimination;

impl IRPass for DeadCodeElimination {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
        let n = ir_graph.stmts.len();
        let ensure_keep: Vec<bool> = ir_graph.stmts.iter().map(|s| s.ir.is_fixed()).collect();
        let (mut _in_d, mut out_d) = ir_graph.get_io_degrees();

        let mut killing_queue: Vec<usize> = Vec::new();
        let mut to_eliminate: Vec<StmtId> = Vec::new();

        for i in 0..n {
            if out_d[i] == 0 && !ensure_keep[i] {
                killing_queue.push(i);
            }
        }

        while let Some(idx) = killing_queue.pop() {
            to_eliminate.push(idx as StmtId);
            for &arg in &ir_graph.stmts[idx].arguments {
                out_d[arg as usize] -= 1;
                if out_d[arg as usize] == 0 && !ensure_keep[arg as usize] {
                    killing_queue.push(arg as usize);
                }
            }
        }

        ir_graph.remove_stmt_bunch(&to_eliminate);
        // P0 seam: signal "something possibly changed" to the resolver.
        // Default StaticOnlyResolver is a no-op; P1's SmtResolver uses
        // this for cache invalidation.
        ir_graph.resolver_mut().on_ir_mutated(&[]);
        ir_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::{CircuitInputs, ResolvedWitness};
    use crate::ir::IRStatement;
    use crate::ir_defs::IR;
    use crate::prove::mock::MockProverBackend;
    use crate::prove::traits::ProverBackend;
    use crate::prove::types::ProvingParams;

    /// `idx = 9 - 4` (a runtime value), then `AssertU128LtBit(idx, hi=4)`
    /// whose returned bit nobody uses.
    fn unused_bit_assert() -> IRGraph {
        let stmts: Vec<(IR, Vec<StmtId>)> = vec![
            (IR::ConstantInt { value: 9 }, vec![]),
            (IR::ConstantInt { value: 4 }, vec![]),
            (IR::SubI, vec![0, 1]),
            (IR::AssertU128LtBit { hi: 4, bit: 0, bits: 2 }, vec![2]),
        ];
        IRGraph::new(
            stmts
                .into_iter()
                .enumerate()
                .map(|(i, (ir, args))| {
                    IRStatement::new(i as StmtId, crate::types::ValueId::next(), ir, args, vec![], None)
                })
                .collect(),
        )
    }

    #[test]
    fn bit_assert_with_unused_bit_keeps_its_range_check() {
        let out = DeadCodeElimination.exec(unused_bit_assert());
        assert!(out.stmts.iter().any(|s| matches!(s.ir, IR::AssertU128LtBit { .. })));
        // idx = 5 is out of [0, 4): the check must still reject it.
        let params = ProvingParams::default();
        let witness = ResolvedWitness::new(CircuitInputs::new(), params.precision_bits);
        let artifact = MockProverBackend.prove(&out, &witness, &params).unwrap();
        assert_ne!(artifact.proof_bytes, "mock_satisfied");
    }
}
