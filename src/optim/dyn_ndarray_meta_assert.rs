use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct DynamicNDArrayMetaAssertInjection;

impl IRPass for DynamicNDArrayMetaAssertInjection {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        // array_id -> (max_rank, max_length)
        let mut meta_lookup: HashMap<u32, (u32, u32)> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();

            let val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, val);

            match &stmt.ir {
                IR::AllocateDynamicNDArrayMeta {
                    array_id,
                    max_rank,
                    max_length,
                    ..
                } => {
                    meta_lookup.insert(*array_id, (*max_rank, *max_length));
                }
                IR::WitnessDynamicNDArrayMeta { array_id, max_rank } => {
                    let (alloc_max_rank, alloc_max_length) = meta_lookup
                        .get(array_id)
                        .unwrap_or_else(|| {
                            panic!(
                                "WitnessDynamicNDArrayMetaIR references unknown array_id={}",
                                array_id
                            )
                        });
                    assert_eq!(
                        *max_rank, *alloc_max_rank,
                        "WitnessDynamicNDArrayMetaIR max_rank mismatch"
                    );
                    // Inject AssertDynamicNDArrayMeta with the same args
                    builder.create_ir(
                        &IR::AssertDynamicNDArrayMeta {
                            array_id: *array_id,
                            max_rank: *alloc_max_rank,
                            max_length: *alloc_max_length,
                        },
                        &ir_args,
                    );
                }
                _ => {}
            }
        }

        builder.export_ir_graph()
    }
}
