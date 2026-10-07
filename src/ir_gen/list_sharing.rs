//! Python lists are shared by reference; Zinnia lists are values.
//!
//! In Python, `b = a`, passing `a` to a function, or storing `a` inside
//! another list or tuple makes a second way to reach the *same* list, so a
//! mutation through one is visible through the other. Zinnia represents a
//! list as an immutable value and rebinds the mutated name to a new value,
//! so the other way would keep the old contents. Rather than compute a
//! different result, a mutation of a list that is reachable through another
//! binding is rejected at compile time.

use crate::types::{Value, ValueId};

use super::IRGenerator;

/// Whether `value` is, or (through lists and tuples) contains, the list or
/// tuple identified by `id`.
fn contains_composite(value: &Value, id: ValueId) -> bool {
    match value {
        Value::List(data) | Value::Tuple(data) => {
            data.value_id == id || data.values.iter().any(|v| contains_composite(v, id))
        }
        _ => false,
    }
}

/// Whether an operation result holds floating-point values.
fn has_float_values(value: &Value) -> bool {
    use crate::types::NumberType;
    match value {
        Value::Float(_) => true,
        Value::StaticArray { dtype, .. } => *dtype == NumberType::Float,
        Value::DynamicNDArray(data) => data.dtype == NumberType::Float,
        Value::List(data) | Value::Tuple(data) => data.values.iter().any(has_float_values),
        _ => false,
    }
}

impl IRGenerator {
    /// Reject mutating `list` (reached through the name `name`) when another
    /// binding can also reach it.
    pub(crate) fn ensure_list_not_shared(&self, name: &str, list: &Value) {
        let Value::List(data) = list else { return };
        let id = data.value_id;
        if let Some(other) = self.ctx.find_other_binding(name, |v| contains_composite(v, id)) {
            panic!(
                "mutating this list through `{}` is not supported: the same list is also \
                 reachable through `{}` (Python shares a list on assignment, argument passing \
                 and storage in another list, but Zinnia copies lists). Make an explicit copy \
                 (`list(...)`) or return the new list instead of mutating it",
                name, other
            );
        }
    }
}

impl IRGenerator {
    /// `name op= value` where `result` is `name op value`. Python updates an
    /// array or a list in place, so every other reference to it sees the
    /// update. Returns `true` when the update was written into the existing
    /// array; `false` when rebinding `name` to `result` is equivalent (no
    /// other binding can observe the old object, or the value is immutable).
    pub(crate) fn update_in_place(&mut self, name: &str, result: &Value) -> bool {
        let Some(current) = self.ctx.get(name) else { return false };
        // Views and detached copies of the array (or of the arrays a list
        // holds) see the update, whether it is written in place or rebinds
        // `name`.
        self.note_array_write(&current);
        match &current {
            Value::StaticArray { dtype, segment_id, .. } => {
                if *dtype != crate::types::NumberType::Float && has_float_values(result) {
                    panic!(
                        "UFuncTypeError: cannot cast the float result of `{} op= ...` into the \
                         integer array `{}` (NumPy's same-kind casting rule)",
                        name, name
                    );
                }
                let segment = *segment_id;
                // The caller sees an in-place kernel's array arguments.
                let shared = self.is_caller_visible_segment(segment)
                    || self
                        .ctx
                        .find_other_binding(name, |v| {
                            super::detached_copies::array_segment(v) == Some(segment)
                        })
                        .is_some();
                if !shared {
                    // The caller rebinds `name` to `result`.
                    self.note_array_moved(&current, result);
                    return false;
                }
                let written = super::named_attr::write_into(&mut self.builder, result, &current);
                self.ctx.set(name, written);
                true
            }
            Value::DynamicNDArray(data) => {
                let segment = data.segment_id;
                if self.is_caller_visible_segment(segment) {
                    panic!(
                        "in-place `{} op= ...` on a bounded-dynamic array argument of a function \
                         that returns no value is not supported",
                        name
                    );
                }
                if let Some(other) = self.ctx.find_other_binding(name, |v| {
                    super::detached_copies::array_segment(v) == Some(segment)
                }) {
                    panic!(
                        "in-place `{} op= ...` on a bounded-dynamic array that is also reachable \
                         through `{}` is not supported",
                        name, other
                    );
                }
                self.note_array_moved(&current, result);
                false
            }
            Value::List(_) => {
                self.ensure_list_not_shared(name, &current);
                self.note_array_moved(&current, result);
                false
            }
            _ => false,
        }
    }
}
