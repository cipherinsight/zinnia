use std::collections::HashMap;
use std::sync::atomic::Ordering;

use crate::ast::*;
use crate::helpers::static_array::base::{
    build_static_array_from_flat, build_static_array_from_flat_complex,
};
use crate::types::{CompositeData, NumberType, Value, ZinniaType};

use super::IRGenerator;

impl IRGenerator {
    pub(crate) fn visit_chip_call(
        &mut self,
        name: &str,
        args: &[Value],
        kwargs: &HashMap<String, Value>,
    ) -> Value {
        let chip = self.registered_chips.get(name).cloned();
        let chip = match chip {
            Some(c) => c,
            None => return Value::None,
        };

        // Parse chip AST
        let chip_ast: ASTNode = match serde_json::from_value(chip.chip_ast.clone()) {
            Ok(node) => node,
            Err(_) => return Value::None,
        };
        let chip_node = match &chip_ast {
            ASTNode::ASTChip(c) => c,
            _ => return Value::None,
        };

        let return_dt = chip
            .return_dt
            .as_ref()
            .map(|dt| self.parse_dt_descriptor(dt));
        let call_condition = self.ctx.get_condition_value(&mut self.builder);

        // A recursive call is unrolled only while the condition under which
        // it is reached may hold. Once that path condition is proved false,
        // the call never runs and is not inlined. A call that is not proved
        // unreachable within `recursion_limit` nested calls is rejected.
        let recursive = self.inlined_chips.iter().any(|inlined| inlined == name);
        if recursive && self.proves_call_unreachable(&call_condition) {
            return self.unreachable_call_value(name, &call_condition, return_dt.as_ref());
        }
        if self.inlined_chips.len() >= self.config.recursion_limit as usize {
            panic!(
                "RecursionLimitExceededError: recursion limit ({}) exceeded while inlining \
                 @zk_chip `{}`: the call is not proven unreachable within that many nested \
                 chip calls. Bound the values its path condition depends on (a bounded array \
                 annotation or a `requires` precondition), or raise ZinniaConfig.recursion_limit \
                 if a larger bound exists",
                self.config.recursion_limit,
                name,
            );
        }

        // Enter chip scope. The call executes only under the caller's
        // current condition; the chip body inherits it as its entry
        // condition so writes, assertions and returns inside the body stay
        // predicated on the call actually happening.
        let entry_condition = (call_condition.bool_val() != Some(true)).then_some(call_condition);
        self.ctx
            .chip_enter(return_dt.clone().unwrap_or(ZinniaType::None), entry_condition);
        self.builder
            .facts
            .enter(crate::optim::predicates::ScopeKind::Chip, None);
        self.inlined_chips.push(name.to_string());

        // Bind arguments. Positional args first, then kwargs by name, then
        // fall back to the chip's ``def f(x, epsilon=1e-6)``-style default
        // expressions for any args that the call site omitted.
        for (i, inp) in chip_node.inputs.iter().enumerate() {
            if i < args.len() {
                self.ctx.set(&inp.name, args[i].clone());
            } else if let Some(kv) = kwargs.get(&inp.name) {
                self.ctx.set(&inp.name, kv.clone());
            } else if let Some(default_node) = &inp.default {
                let default_value = self.visit(default_node);
                self.ctx.set(&inp.name, default_value);
            }
        }

        self.register_global_datatypes();

        // Visit chip body
        for stmt in &chip_node.block {
            self.visit(stmt);
        }

        let return_guaranteed = self.ctx.check_return_guaranteed();

        // Collect return value
        // Collect returns BEFORE leaving chip scope
        let returns = self.ctx.get_returns_with_conditions();

        // Check return guarantee: error if chip has non-None return type
        // and no return statement was encountered on any path
        if return_dt
            .as_ref()
            .is_some_and(|dt| !matches!(dt, ZinniaType::None))
            && returns.is_empty()
        {
            panic!("Chip control ends without a return statement");
        }
        if !returns.is_empty() && !return_guaranteed {
            if return_dt.is_some() {
                panic!("Chip return annotation requires every path to return");
            }
            panic!(
                "Chip return inference does not support partial returns without an explicit return annotation"
            );
        }
        if self.ctx.has_mixed_bare_and_value_returns() {
            panic!("Chip return inference does not allow mixing bare returns with value returns");
        }

        self.ctx.chip_leave();
        let _chip_facts = self.builder.facts.leave();
        self.inlined_chips.pop();

        // Merge return values using conditional select
        if returns.is_empty() {
            return Value::None;
        }
        let mut result = returns[0].0.clone();
        for i in 1..returns.len() {
            let (val, cond) = &returns[i];
            // A helper returns one of its argument arrays itself in Python.
            result = self.merge_values(cond, val, &result);
        }
        if let Some(return_dt) = return_dt {
            if result.zinnia_type() != return_dt {
                result = self.try_cast_value_to_match(result, &return_dt);
            }
            if result.zinnia_type() != return_dt {
                if let (ZinniaType::NDArray { shape, dtype }, Value::List(_) | Value::Tuple(_)) =
                    (&return_dt, &result)
                {
                    if self.composite_matches_ndarray_return(&result, shape, *dtype) {
                        return result;
                    }
                }
                panic!(
                    "Chip return type mismatch: expected {}, got {}",
                    return_dt,
                    result.zinnia_type()
                );
            }
        }
        result
    }

    /// Whether the compiler proves that a call reached under `condition`
    /// never runs: a constant, the proven bounds of the compared values (as
    /// for a `while` guard, `provably_false`), then the decision query
    /// `condition <= 0` (`IRBuilder::proves_at_most`: intervals, scalar and
    /// array facts, then one Z3 check within its step budget, assuming the
    /// program's preconditions). A decision query, unlike `resolve_bool`,
    /// caches nothing about `condition` as unknown, so the next, deeper call
    /// is still encoded with the full definition of this one's condition.
    fn proves_call_unreachable(&mut self, condition: &Value) -> bool {
        if let Some(holds) = condition.bool_val().or_else(|| condition.int_val().map(|v| v != 0)) {
            return !holds;
        }
        if self.provably_false(condition) {
            return true;
        }
        self.builder.proves_at_most(condition, 0)
    }

    /// The value of a recursive call to `name` that never runs. No
    /// execution observes it, so any value of the chip's return type serves:
    /// zero. When the conclusion came from reasoning rather than a constant,
    /// the circuit also asserts that the call is not reached, so a wrong
    /// conclusion makes the circuit unsatisfiable instead of dropping the
    /// call (as for a `while` loop, `assert_while_exhausted`).
    fn unreachable_call_value(
        &mut self,
        name: &str,
        condition: &Value,
        return_dt: Option<&ZinniaType>,
    ) -> Value {
        if let Some(t) = self.builder.resolver_telemetry() {
            t.recursive_calls_proved_unreachable.fetch_add(1, Ordering::Relaxed);
        }
        if condition.bool_val().is_none() {
            let unreached = self.builder.ir_logical_not(condition);
            self.assert_value(&unreached);
        }
        // Without an annotation, the chip's type is that of the values the
        // calling frame (the same chip, for direct recursion) returned so far.
        let inferred = || -> Option<ZinniaType> {
            if self.inlined_chips.last().map(String::as_str) != Some(name) {
                return None;
            }
            let returns = self.ctx.get_returns_with_conditions();
            returns.first().map(|(value, _)| value.zinnia_type())
        };
        let return_type = return_dt.cloned().or_else(inferred);
        return_type
            .and_then(|ty| self.zero_of_type(&ty))
            .unwrap_or_else(|| {
                panic!(
                    "the recursive call to @zk_chip `{}` is proven unreachable, but its result \
                     has no known type to stand in for it. Annotate the chip's return type \
                     (a scalar, a fixed-shape array, or a tuple or list of them)",
                    name
                )
            })
    }

    /// The zero value of a scalar, fixed-shape array, or tuple or list of
    /// them; `None` for any other type.
    fn zero_of_type(&mut self, ty: &ZinniaType) -> Option<Value> {
        let b = &mut self.builder;
        Some(match ty {
            ZinniaType::None => Value::None,
            ZinniaType::Integer => b.ir_constant_int(0),
            ZinniaType::Float => b.ir_constant_float(0.0),
            ZinniaType::Boolean => b.ir_constant_bool(false),
            ZinniaType::Complex => crate::helpers::value_ops::pack_complex_value(
                b.ir_constant_float(0.0),
                b.ir_constant_float(0.0),
            ),
            ZinniaType::NDArray { shape, dtype } => {
                let cells = shape.iter().product::<usize>();
                let shape = shape.clone();
                match dtype {
                    NumberType::Complex => {
                        let zeros = vec![b.ir_constant_float(0.0); cells];
                        build_static_array_from_flat_complex(b, zeros.clone(), zeros, shape)
                    }
                    NumberType::Float => {
                        let zeros = vec![b.ir_constant_float(0.0); cells];
                        build_static_array_from_flat(b, zeros, shape, *dtype)
                    }
                    NumberType::Integer => {
                        let zeros = vec![b.ir_constant_int(0); cells];
                        build_static_array_from_flat(b, zeros, shape, *dtype)
                    }
                }
            }
            ZinniaType::Tuple { elements } | ZinniaType::List { elements } => {
                let values = elements
                    .iter()
                    .map(|element| self.zero_of_type(element))
                    .collect::<Option<Vec<_>>>()?;
                let data = CompositeData::new(elements.clone(), values);
                if matches!(ty, ZinniaType::Tuple { .. }) {
                    Value::Tuple(data)
                } else {
                    Value::List(data)
                }
            }
            _ => return None,
        })
    }

    fn composite_matches_ndarray_return(
        &self,
        value: &Value,
        shape: &[usize],
        dtype: NumberType,
    ) -> bool {
        if crate::helpers::composite::get_composite_shape(value) != shape {
            return false;
        }
        crate::helpers::composite::flatten_composite(value)
            .iter()
            .all(|leaf| match (dtype, leaf) {
                (NumberType::Integer, Value::Integer(_)) => true,
                (NumberType::Float, Value::Float(_)) => true,
                (NumberType::Complex, Value::Complex { .. }) => true,
                _ => false,
            })
    }

    pub(crate) fn visit_external_call(&mut self, name: &str, args: &[Value]) -> Value {
        let ext = self.registered_externals.get(name).cloned();
        let ext = match ext {
            Some(e) => e,
            None => return Value::None,
        };

        let return_dt = self.parse_dt_descriptor(&ext.return_dt);

        // Build arg type descriptors for InvokeExternal
        let arg_dts: Vec<serde_json::Value> = args
            .iter()
            .map(|a| match a.zinnia_type() {
                ZinniaType::Integer | ZinniaType::Boolean => {
                    serde_json::json!({"__class__": "IntegerDTDescriptor", "dt_data": {}})
                }
                ZinniaType::Float => {
                    serde_json::json!({"__class__": "FloatDTDescriptor", "dt_data": {}})
                }
                _ => serde_json::json!({"__class__": "IntegerDTDescriptor", "dt_data": {}}),
            })
            .collect();

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
