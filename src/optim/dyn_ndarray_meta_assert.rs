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
        // array_id -> (max_rank, max_length, max_shape)
        let mut meta_lookup: HashMap<u32, (u32, u32, Vec<u32>)> = HashMap::new();

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
                    max_shape,
                    ..
                } => {
                    meta_lookup.insert(*array_id, (*max_rank, *max_length, max_shape.clone()));
                }
                IR::WitnessDynamicNDArrayMeta {
                    array_id, max_rank, ..
                } => {
                    let (alloc_max_rank, alloc_max_length, alloc_max_shape) =
                        meta_lookup.get(array_id).unwrap_or_else(|| {
                            panic!(
                                "WitnessDynamicNDArrayMetaIR references unknown array_id={}",
                                array_id
                            )
                        });
                    assert_eq!(
                        *max_rank, *alloc_max_rank,
                        "WitnessDynamicNDArrayMetaIR max_rank mismatch"
                    );
                    let mut assert_args = ir_args.clone();
                    for axis_bound in alloc_max_shape {
                        let bound_value = builder.ir_constant_int(*axis_bound as i64);
                        assert_args.push(bound_value);
                    }
                    // Inject AssertDynamicNDArrayMeta with runtime shape args,
                    // followed by per-axis maxima from allocation metadata.
                    builder.create_ir(
                        &IR::AssertDynamicNDArrayMeta {
                            array_id: *array_id,
                            max_rank: *alloc_max_rank,
                            max_length: *alloc_max_length,
                        },
                        &assert_args,
                    );
                }
                _ => {}
            }
        }

        let mut new_graph = builder.export_ir_graph();
        // P0 seam: signal mutation to the resolver. No-op today via
        // StaticOnlyResolver; P1 uses this for cache invalidation.
        new_graph.resolver_mut().on_ir_mutated(&[]);
        new_graph
    }
}
