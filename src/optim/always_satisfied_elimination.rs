use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct AlwaysSatisfiedElimination;

impl IRPass for AlwaysSatisfiedElimination {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
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

        // Find assertions that are always satisfied
        let mut to_eliminate: Vec<StmtId> = Vec::new();
        for stmt in &ir_graph.stmts {
            if matches!(stmt.ir, IR::Assert) {
                let cond_ptr = stmt.arguments[0];
                if let Some(cond_val) = &values_lookup.get(&cond_ptr) {
                    if let Some(v) = cond_val.int_val() {
                        if v != 0 {
                            to_eliminate.push(stmt.stmt_id);
                        }
                    }
                }
            }
        }

        ir_graph.remove_stmt_bunch(&to_eliminate);
        ir_graph
    }
}
