use std::collections::HashMap;
use std::sync::atomic::Ordering;

use crate::ast::*;
use crate::types::{Value, ZinniaType};

use super::{ChipCallFrame, IRGenerator};

impl IRGenerator {
    pub(crate) fn visit_chip_call(&mut self, name: &str, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
        let chip = self.registered_chips.get(name).cloned();
        let chip = match chip {
            Some(c) => c,
            None => return Value::None,
        };

        // P4 round 2 — recursive-chip bound discharge.
        //
        // Find the most recent prior frame for the same chip name. If
        // present, this call is recursive into `name` and we have its
        // parent's argument bindings available for the heuristic; if
        // absent, this is a non-recursive entry — keep today's
        // behaviour (full `recursion_limit` budget for the new frame).
        //
        // The heuristic picks the integer arg with the most-negative
        // delta (`args[i].int_val() - parent.int_args[i]`). Tie-break
        // by lowest index for determinism. If no integer arg strictly
        // decreases, no measure is selectable and we fall through to
        // the global `recursion_limit` budget — telemetry's
        // `recursion_no_measure_found` counter surfaces those calls.
        //
        // SMT-resolved bounds only ever **tighten** the new frame's
        // forward-looking depth allowance (`remaining_bound`); they
        // never loosen it past `recursion_limit`. The check fires when
        // a recursive call is attempted but the parent's
        // `remaining_bound` is already zero. This preserves round-1's
        // safety net against exponentially-branching recursions
        // (e.g. naive fibonacci) regardless of what the resolver
        // returns.
        let parent_frame_idx = self
            .chip_call_stack
            .iter()
            .rposition(|f| f.chip_name == name);

        let new_frame_bound: u32 = if let Some(idx) = parent_frame_idx {
            let parent_remaining = self.chip_call_stack[idx].remaining_bound;
            // The parent must permit at least one more descent. Without
            // this check a misbehaving heuristic could let recursion
            // descend past today's safety net.
            if parent_remaining == 0 {
                let cited = self.config.recursion_limit;
                panic!(
                    "RecursionLimitExceededError: recursion limit ({}) exceeded while inlining @zk_chip `{}`. \
                    Each recursive call into a chip is unrolled at compile time; if your chip recurses with \
                    multiple call sites per level (e.g. fibo(n-1) + fibo(n-2)), the unrolled depth grows \
                    exponentially. Either reduce the recursion in the chip body or raise \
                    ZinniaConfig.recursion_limit (current default is intentionally small).",
                    cited,
                    name,
                );
            }

            // Pick the recursion measure: the integer arg with the
            // most-negative delta vs the parent's binding. Args at the
            // recursive call site are `args[i]`; parent's bindings are
            // `chip_call_stack[idx].int_args[i]`. Only positions where
            // both sides have a compile-time `int_val()` are considered
            // for the delta comparison; the picked measure's value is
            // the one we hand to `resolve_max`.
            let parent_int_args = self.chip_call_stack[idx].int_args.clone();
            let mut best: Option<(usize, i64, Value)> = None; // (i, delta, child_val)
            for (i, child_val) in args.iter().enumerate() {
                let child_int = match child_val.int_val() {
                    Some(v) => v,
                    None => continue,
                };
                let parent_int = match parent_int_args.get(i).and_then(|p| *p) {
                    Some(v) => v,
                    None => continue,
                };
                let delta = child_int - parent_int;
                if delta >= 0 {
                    continue;
                }
                let take = match &best {
                    None => true,
                    Some((_bi, bd, _bv)) => delta < *bd,
                };
                if take {
                    best = Some((i, delta, child_val.clone()));
                }
            }

            // Resolve the measure to an upper bound, with fast-path
            // discipline (round-1.5 lesson): `int_val()` first
            // (zero-cost); only consult the layered resolver when the
            // value is non-trivially symbolic.
            let measure_bound: Option<u32> = match best {
                Some((_i, _d, ref measure)) => {
                    if let Some(n) = measure.int_val() {
                        if let Some(t) = self.builder.resolver_telemetry() {
                            t.recursion_bound_static_val.fetch_add(1, Ordering::Relaxed);
                        }
                        Some(n.max(0) as u32)
                    } else {
                        let (resolver, stmts) = self.builder.split_resolver_and_stmts();
                        let resolved = resolver.resolve_max_with_stmts(measure, stmts);
                        if let Some(n) = resolved {
                            if let Some(t) = self.builder.resolver_telemetry() {
                                t.recursion_bound_resolver_proved
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            Some(n.max(0) as u32)
                        } else {
                            None
                        }
                    }
                }
                None => {
                    if let Some(t) = self.builder.resolver_telemetry() {
                        t.recursion_no_measure_found.fetch_add(1, Ordering::Relaxed);
                    }
                    None
                }
            };

            // Forward-looking budget for the new frame: at most
            // `parent_remaining - 1` (we just consumed one level of the
            // parent's allowance) AND at most `measure_bound` (an upper
            // bound on the measure ⇒ at most that many further
            // descents). Either side may be looser; we pick the tighter
            // one. SMT-resolved bound only ever tightens — never
            // loosens past `recursion_limit`.
            let from_parent = parent_remaining.saturating_sub(1);
            match measure_bound {
                Some(b) => from_parent.min(b).min(self.config.recursion_limit),
                None => from_parent.min(self.config.recursion_limit),
            }
        } else {
            // Non-recursive entry — fresh budget at today's safety net.
            self.config.recursion_limit
        };

        // Check the global recursion-depth safety net. This is today's
        // hard cap; the round-2 wiring layered above only ever
        // tightens, never loosens. So this check is the floor.
        if self.recursion_depth >= self.config.recursion_limit {
            panic!(
                "RecursionLimitExceededError: recursion limit ({}) exceeded while inlining @zk_chip `{}`. \
                Each recursive call into a chip is unrolled at compile time; if your chip recurses with \
                multiple call sites per level (e.g. fibo(n-1) + fibo(n-2)), the unrolled depth grows \
                exponentially. Either reduce the recursion in the chip body or raise \
                ZinniaConfig.recursion_limit (current default is intentionally small).",
                self.config.recursion_limit,
                name,
            );
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

        // P4 round 2 — push our frame onto the chip-call stack so any
        // recursive call from inside this body can diff against our
        // bindings. Snapshot only the integer-value-known args; non-int
        // and unknown-int args become `None` slots (the heuristic
        // ignores them).
        let int_args: Vec<Option<i64>> = chip_node
            .inputs
            .iter()
            .map(|inp| self.ctx.get(&inp.name).and_then(|v| v.int_val()))
            .collect();
        self.chip_call_stack.push(ChipCallFrame {
            chip_name: name.to_string(),
            int_args,
            remaining_bound: new_frame_bound,
        });

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
        self.chip_call_stack.pop();

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
