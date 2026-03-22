use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct DynamicNDArrayMemoryLowering {
    pub mux_threshold: u32,
}

impl DynamicNDArrayMemoryLowering {
    pub fn new(mux_threshold: u32) -> Self {
        assert!(mux_threshold > 0, "mux_threshold must be positive");
        Self { mux_threshold }
    }
}

impl IRPass for DynamicNDArrayMemoryLowering {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        // Collect segment sizes and which segments have dynamic access
        let mut segment_sizes: HashMap<u32, u32> = HashMap::new();
        let mut dynamic_access_segments: HashSet<u32> = HashSet::new();

        for stmt in &ir_graph.stmts {
            if let IR::AllocateMemory { segment_id, size, .. } = &stmt.ir {
                segment_sizes.insert(*segment_id, *size);
            }
            match &stmt.ir {
                IR::DynamicNDArrayGetItem { segment_id, .. }
                | IR::DynamicNDArraySetItem { segment_id, .. } => {
                    dynamic_access_segments.insert(*segment_id);
                }
                _ => {}
            }
        }

        let mux_segments: HashSet<u32> = segment_sizes
            .iter()
            .filter(|(seg_id, &seg_size)| {
                dynamic_access_segments.contains(seg_id) && seg_size < self.mux_threshold
            })
            .map(|(&seg_id, _)| seg_id)
            .collect();

        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut mux_segment_cells: HashMap<u32, Vec<Value>> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();

            match &stmt.ir {
                IR::AllocateMemory {
                    segment_id,
                    size,
                    init_value,
                } if mux_segments.contains(segment_id) => {
                    let init_val = builder.ir_constant_int(*init_value);
                    let cells: Vec<Value> = (0..*size).map(|_| init_val.clone()).collect();
                    mux_segment_cells.insert(*segment_id, cells);
                    value_lookup.insert(stmt.stmt_id, Value::None);
                    continue;
                }

                IR::DynamicNDArrayGetItem {
                    segment_id,
                    array_id: _,
                } => {
                    if mux_segments.contains(segment_id) {
                        if let Some(cells) = mux_segment_cells.get(segment_id) {
                            let idx_val = &ir_args[0];
                            let mut lowered = cells[0].clone();
                            for (i, cell) in cells.iter().enumerate() {
                                let ci = builder.ir_constant_int(i as i64);
                                let cond = builder.ir_equal_i(idx_val, &ci);
                                lowered = builder.ir_select_i(&cond, cell, &lowered);
                            }
                            value_lookup.insert(stmt.stmt_id, lowered);
                            continue;
                        }
                    }
                    // Fall through to ReadMemory lowering
                    let lowered = builder.create_ir(
                        &IR::ReadMemory {
                            segment_id: *segment_id,
                        },
                        &[ir_args[0].clone()],
                    );
                    value_lookup.insert(stmt.stmt_id, lowered);
                    continue;
                }

                IR::DynamicNDArraySetItem {
                    segment_id,
                    array_id: _,
                } => {
                    if mux_segments.contains(segment_id) {
                        if let Some(cells) = mux_segment_cells.get(segment_id).cloned() {
                            let idx_val = &ir_args[0];
                            let write_val = &ir_args[1];
                            let mut next_cells = Vec::new();
                            for (i, cell) in cells.iter().enumerate() {
                                let ci = builder.ir_constant_int(i as i64);
                                let cond = builder.ir_equal_i(idx_val, &ci);
                                next_cells.push(builder.ir_select_i(&cond, write_val, cell));
                            }
                            mux_segment_cells.insert(*segment_id, next_cells);
                            value_lookup.insert(stmt.stmt_id, Value::None);
                            continue;
                        }
                    }
                    // Fall through to WriteMemory lowering
                    let lowered = builder.create_ir(
                        &IR::WriteMemory {
                            segment_id: *segment_id,
                        },
                        &[ir_args[0].clone(), ir_args[1].clone()],
                    );
                    value_lookup.insert(stmt.stmt_id, lowered);
                    continue;
                }

                _ => {}
            }

            let val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, val);
        }

        builder.export_ir_graph()
    }
}
