use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::StmtId;

use super::IRPass;

pub struct ExternalCallRemover;

impl IRPass for ExternalCallRemover {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
        let to_remove: Vec<StmtId> = ir_graph
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::InvokeExternal { .. } | IR::ExportExternalI { .. } | IR::ExportExternalF { .. }))
            .map(|s| s.stmt_id)
            .collect();
        ir_graph.remove_stmt_bunch(&to_remove);
        ir_graph
    }
}
