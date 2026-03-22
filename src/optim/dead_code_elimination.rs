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
        ir_graph
    }
}
