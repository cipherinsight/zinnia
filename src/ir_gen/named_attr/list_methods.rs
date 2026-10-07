//! Dispatch for Python list methods invoked as `var.method(...)` where the
//! target is a known variable. The actual method bodies live in
//! `src/ir_gen/list_methods.rs` (sibling module) — this file only routes
//! the (target, member) pair to the right helper.

use crate::types::Value;

use super::super::IRGenerator;
use super::DispatchCtx;

impl IRGenerator {
    pub(crate) fn try_list_method(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        let var = ctx.target?;
        if !self.ctx.exists(var) {
            return None;
        }
        // P4a: list literals arrive as segment-backed StaticArrays, but
        // every method body below pattern-matches `Value::List` — without
        // materialisation they silently no-op (`lst.reverse()` left the
        // list unchanged). Rebind as a List first; these are Python-list
        // operations, so keeping the segment backing has no value.
        if matches!(
            ctx.member,
            "append"
                | "extend"
                | "pop"
                | "remove"
                | "clear"
                | "reverse"
                | "count"
                | "index"
                | "insert"
        ) {
            if let Some(v @ Value::StaticArray { .. }) = self.ctx.get(var) {
                let lst = crate::helpers::static_array::base::to_value_list(&mut self.builder, &v);
                self.ctx.set(var, lst);
            }
        }
        if matches!(
            ctx.member,
            "append" | "extend" | "pop" | "remove" | "clear" | "reverse" | "insert"
        ) {
            if self.expression_guard_depth > 0 {
                panic!(
                    "`{}.{}(...)` inside a comprehension filter or one arm of a conditional \
                     expression is not supported: the change would happen on some paths only, \
                     and Zinnia does not carry it out of the expression. Write a `for` loop \
                     with an `if` statement instead",
                    var, ctx.member
                );
            }
            if let Some(list) = self.ctx.get(var) {
                self.ensure_list_not_shared(var, &list);
            }
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
                match &v {
                    Value::StaticArray { .. } => {
                        let lst = crate::helpers::static_array::base::to_value_list(
                            &mut self.builder,
                            &v,
                        );
                        crate::helpers::static_array::base::to_static_array(&mut self.builder, &lst)
                            .unwrap_or(lst)
                    }
                    // `lst.copy()` is a new (shallow) list, not `lst` itself.
                    Value::List(data) => Value::List(crate::types::CompositeData {
                        value_id: crate::types::ValueId::next(),
                        ..data.clone()
                    }),
                    // A runtime-shaped array's copy is a new array, as for
                    // a static one (it used to be the same array, so writes
                    // through either reached the other).
                    Value::DynamicNDArray(_) => super::array_compat::copy_array(&mut self.builder, &v),
                    _ => v,
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
