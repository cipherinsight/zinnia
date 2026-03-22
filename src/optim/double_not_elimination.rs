use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct DoubleNotElimination;

impl IRPass for DoubleNotElimination {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        // Maps ptr of a NOT result -> the original operand value
        let mut not_original: HashMap<StmtId, Value> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();

            let result = if matches!(stmt.ir, IR::LogicalNot) {
                let operand = &ir_args[0];
                if let Some(ptr) = operand.ptr() {
                    if let Some(orig) = not_original.get(&ptr) {
                        // Double negation — eliminate
                        orig.clone()
                    } else {
                        let new_val = builder.create_ir(&stmt.ir, &ir_args);
                        if let Some(new_ptr) = new_val.ptr() {
                            not_original.insert(new_ptr, operand.clone());
                        }
                        new_val
                    }
                } else {
                    builder.create_ir(&stmt.ir, &ir_args)
                }
            } else {
                builder.create_ir(&stmt.ir, &ir_args)
            };

            value_lookup.insert(stmt.stmt_id, result);
        }

        builder.export_ir_graph()
    }
}
