use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct ConstantFold;

impl IRPass for ConstantFold {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut constant_int_cache: HashMap<i64, Value> = HashMap::new();
        let mut constant_float_cache: HashMap<u64, Value> = HashMap::new(); // f64 bits as key
        let constant_true = builder.ir_constant_bool(true);
        let constant_false = builder.ir_constant_bool(false);

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| {
                    let value = value_lookup[&arg].clone();
                    // Replace known constants with cached constant IRs
                    match &value {
                        Value::Boolean(sv) => match sv.static_val {
                            Some(true) => constant_true.clone(),
                            Some(false) => constant_false.clone(),
                            None => value,
                        },
                        Value::Integer(sv) => {
                            if let Some(v) = sv.static_val {
                                constant_int_cache
                                    .entry(v)
                                    .or_insert_with(|| builder.ir_constant_int(v))
                                    .clone()
                            } else {
                                value
                            }
                        }
                        Value::Float(sv) => {
                            if let Some(v) = sv.static_val {
                                let bits = v.to_bits();
                                constant_float_cache
                                    .entry(bits)
                                    .or_insert_with(|| builder.ir_constant_float(v))
                                    .clone()
                            } else {
                                value
                            }
                        }
                        _ => value,
                    }
                })
                .collect();

            let new_val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, new_val);
        }

        builder.export_ir_graph()
    }
}
