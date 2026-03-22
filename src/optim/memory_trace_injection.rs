use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct MemoryTraceInjection;

impl IRPass for MemoryTraceInjection {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut has_memory_access = false;

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();

            let new_val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, new_val.clone());

            match &stmt.ir {
                IR::WriteMemory { segment_id } => {
                    has_memory_access = true;
                    builder.create_ir(
                        &IR::MemoryTraceEmit {
                            segment_id: *segment_id,
                            is_write: true,
                        },
                        &[ir_args[0].clone(), ir_args[1].clone()],
                    );
                }
                IR::ReadMemory { segment_id } => {
                    has_memory_access = true;
                    builder.create_ir(
                        &IR::MemoryTraceEmit {
                            segment_id: *segment_id,
                            is_write: false,
                        },
                        &[ir_args[0].clone(), new_val],
                    );
                }
                _ => {}
            }
        }

        if has_memory_access {
            builder.create_ir(&IR::MemoryTraceSeal, &[]);
        }

        builder.export_ir_graph()
    }
}
