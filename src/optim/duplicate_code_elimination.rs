use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct DuplicateCodeElimination;

impl IRPass for DuplicateCodeElimination {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        // Phase 1: identify duplicates
        let mut to_be_replaced: HashMap<StmtId, StmtId> = HashMap::new();
        let mut seen: Vec<(IR, Vec<StmtId>, StmtId)> = Vec::new();

        for stmt in &ir_graph.stmts {
            let mut existing = None;
            for (ir, args, id) in &seen {
                if *ir == stmt.ir && *args == stmt.arguments {
                    existing = Some(*id);
                    break;
                }
            }
            if let Some(existing_id) = existing {
                to_be_replaced.insert(stmt.stmt_id, existing_id);
            } else {
                seen.push((stmt.ir.clone(), stmt.arguments.clone(), stmt.stmt_id));
            }
        }

        // Phase 2: rebuild graph, replacing duplicates
        // values_lookup maps original stmt_id -> Value from the new builder
        // For duplicated stmts, we resolve through to_be_replaced to the canonical stmt
        let mut builder = IRBuilder::new();
        let mut values_lookup: HashMap<StmtId, Value> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            if let Some(&replacement) = to_be_replaced.get(&stmt.stmt_id) {
                // This stmt is a duplicate; point it at the canonical stmt's value
                // The canonical stmt must already be in values_lookup
                let val = values_lookup[&replacement].clone();
                values_lookup.insert(stmt.stmt_id, val);
                continue;
            }

            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| {
                    let resolved = to_be_replaced.get(&arg).copied().unwrap_or(arg);
                    values_lookup[&resolved].clone()
                })
                .collect();

            let val = builder.create_ir(&stmt.ir, &ir_args);
            values_lookup.insert(stmt.stmt_id, val);
        }

        builder.export_ir_graph()
    }
}
