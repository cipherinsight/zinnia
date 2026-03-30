use std::collections::HashMap;

use crate::ast::*;
use crate::types::{Value, ZinniaType};

use super::IRGenerator;

impl IRGenerator {
    pub(crate) fn visit_chip_call(&mut self, name: &str, args: &[Value], _kwargs: &HashMap<String, Value>) -> Value {
        let chip = self.registered_chips.get(name).cloned();
        let chip = match chip {
            Some(c) => c,
            None => return Value::None,
        };

        // Check recursion limit
        if self.recursion_depth >= self.config.recursion_limit {
            // Return a placeholder value
            let return_dt = self.parse_dt_descriptor(&chip.return_dt);
            return match return_dt {
                ZinniaType::Integer | ZinniaType::Boolean => self.builder.ir_constant_int(0),
                ZinniaType::Float => self.builder.ir_constant_float(0.0),
                _ => Value::None,
            };
        }

        // Parse chip AST
        let chip_ast: ASTNode = match serde_json::from_value(chip.chip_ast.clone()) {
            Ok(node) => node,
            Err(_) => return Value::None,
        };
        let chip_node = match &chip_ast {
            ASTNode::ASTChip(c) => c,
            _ => return Value::None,
        };

        // Enter chip scope
        let return_dt = self.parse_dt_descriptor(&chip.return_dt);
        self.ctx.chip_enter(return_dt, None);
        self.recursion_depth += 1;

        // Bind arguments
        for (i, inp) in chip_node.inputs.iter().enumerate() {
            if i < args.len() {
                self.ctx.set(&inp.name, args[i].clone());
            }
        }

        self.register_global_datatypes();

        // Visit chip body
        for stmt in &chip_node.block {
            self.visit(stmt);
        }

        // Check if return is guaranteed for non-None return types
        let return_dt_check = self.parse_dt_descriptor(&chip.return_dt);
        let return_guaranteed = self.ctx.check_return_guaranteed();

        // Collect return value
        // Collect returns BEFORE leaving chip scope
        let returns = self.ctx.get_returns_with_conditions();

        // Check return guarantee: error if chip has non-None return type
        // and no return statement was encountered on any path
        if !matches!(return_dt_check, ZinniaType::None) && returns.is_empty() {
            panic!("Chip control ends without a return statement");
        }

        self.ctx.chip_leave();
        self.recursion_depth -= 1;

        // Merge return values using conditional select
        if returns.is_empty() {
            return Value::None;
        }
        let mut result = returns[0].0.clone();
        for i in 1..returns.len() {
            let (val, cond) = &returns[i];
            result = crate::helpers::value_ops::select_value(&mut self.builder, cond, val, &result);
        }
        result
    }

    pub(crate) fn visit_external_call(&mut self, name: &str, args: &[Value]) -> Value {
        let ext = self.registered_externals.get(name).cloned();
        let ext = match ext {
            Some(e) => e,
            None => return Value::None,
        };

        let return_dt = self.parse_dt_descriptor(&ext.return_dt);

        // Build arg type descriptors for InvokeExternal
        let arg_dts: Vec<serde_json::Value> = args.iter().map(|a| {
            match a.zinnia_type() {
                ZinniaType::Integer | ZinniaType::Boolean => serde_json::json!({"__class__": "IntegerDTDescriptor", "dt_data": {}}),
                ZinniaType::Float => serde_json::json!({"__class__": "FloatDTDescriptor", "dt_data": {}}),
                _ => serde_json::json!({"__class__": "IntegerDTDescriptor", "dt_data": {}}),
            }
        }).collect();

        // Allocate a unique store index for this external call
        let store_idx = self.next_external_store_idx;
        self.next_external_store_idx += 1;

        // Export each argument
        for (i, arg) in args.iter().enumerate() {
            let flat = crate::helpers::composite::flatten_composite(arg);
            for (j, v) in flat.iter().enumerate() {
                let key = crate::ir_defs::ExternalKey::Int(j as u32);
                match v {
                    Value::Float(_) => {
                        self.builder.create_ir(
                            &crate::ir_defs::IR::ExportExternalF {
                                for_which: store_idx,
                                key,
                                indices: vec![i as u32],
                            },
                            &[v.clone()],
                        );
                    }
                    _ => {
                        self.builder.create_ir(
                            &crate::ir_defs::IR::ExportExternalI {
                                for_which: store_idx,
                                key,
                                indices: vec![i as u32],
                            },
                            &[v.clone()],
                        );
                    }
                }
            }
        }

        // Invoke the external function
        let invoke_ir = crate::ir_defs::IR::InvokeExternal {
            store_idx,
            func_name: name.to_string(),
            args: arg_dts,
            kwargs: std::collections::HashMap::new(),
        };
        self.builder.create_ir(&invoke_ir, &[]);

        // Read the external function result (resolved during preprocessing).
        let is_float = matches!(return_dt, ZinniaType::Float);
        self.builder.ir_read_external_result(store_idx, 0, is_float)
    }
}
