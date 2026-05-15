//! Dispatch for Python list methods invoked as `var.method(...)` where the
//! target is a known variable. The actual method bodies live in
//! `src/ir_gen/list_methods.rs` (sibling module) — this file only routes
//! the (target, member) pair to the right helper.

use crate::types::Value;

use super::DispatchCtx;
use super::super::IRGenerator;

impl IRGenerator {
    pub(crate) fn try_list_method(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        let var = ctx.target?;
        if !self.ctx.exists(var) {
            return None;
        }
        let visited_args = ctx.args;
        let v = match ctx.member {
            "append" => self.list_method_append(var, visited_args),
            "extend" => self.list_method_extend(var, visited_args),
            "pop" => self.list_method_pop(var, visited_args),
            "remove" => self.list_method_remove(var, visited_args),
            "clear" => self.list_method_clear(var),
            "copy" => {
                let v = self.ctx.get(var).unwrap_or(Value::None);
                // For Value::StaticArray, .copy() must clone the underlying
                // segment so subsequent mutations don't alias back to the
                // source. Without this, P3 segment-shared writes would
                // mutate both the original and the copy.
                if let Value::StaticArray { .. } = &v {
                    let lst = crate::helpers::static_array::base::to_value_list(&mut self.builder, &v);
                    if let Some(sa) = crate::helpers::static_array::base::to_static_array(&mut self.builder, &lst) {
                        sa
                    } else {
                        lst
                    }
                } else {
                    v
                }
            }
            "reverse" => self.list_method_reverse(var),
            "count" => self.list_method_count(var, visited_args),
            "index" => self.list_method_index(var, visited_args),
            "insert" => self.list_method_insert(var, visited_args),
            _ => return None,
        };
        Some(v)
    }
}
