//! NumPy's `bool` dtype on arrays.
//!
//! Zinnia stores a bool array like an integer array (`NumberType::Integer`,
//! cells 0 and 1). NumPy tells them apart by dtype, and indexing depends on
//! it: `a[m]` keeps the elements where a bool `m` is true, but gathers the
//! elements at the positions an integer `m` lists
//! (`helpers::array_ops::array_index_kind`). The builder records the dtype
//! for cells and for array storage:
//!
//! * a **bool cell** is a `Value::Boolean`, or an integer-typed value that
//!   holds a NumPy bool: a read from a bool segment (`ir_read_memory`), or a
//!   select between two bool cells (`value_ops::select_value`);
//! * a **bool segment** stores a bool array. A segment becomes one when it is
//!   created from bool cells only (`segment::alloc_and_write_values`,
//!   `static_array::base::build_static_array_from_flat`), or when an
//!   operation whose NumPy result is a bool array marks it
//!   ([`IRBuilder::mark_bool_segment`]: a boolean mask applied to a bool
//!   array, a sort of one, `bool` input parameters). A view shares its base's
//!   segment and so its dtype. Writes do not change a segment's dtype.
//!
//! Recording a dtype emits no IR; it only decides how an array used as an
//! index is read.

use crate::types::Value;

use super::IRBuilder;

impl IRBuilder {
    /// Whether `v` holds a NumPy bool (see the module documentation).
    pub fn is_bool_cell(&self, v: &Value) -> bool {
        match v {
            Value::Boolean(_) => true,
            Value::Integer(s) => s.stmt_id.is_some_and(|sid| self.bool_cells.contains(&sid)),
            _ => false,
        }
    }

    /// Record that the integer-typed runtime value `v` holds a NumPy bool.
    /// Only for values whose every possible value is 0 or 1 and whose NumPy
    /// dtype is bool.
    pub fn record_bool_cell(&mut self, v: &Value) {
        if let Some(sid) = v.stmt_id() {
            self.bool_cells.insert(sid);
        }
    }

    /// Whether `segment_id` stores a NumPy bool array.
    pub fn is_bool_segment(&self, segment_id: u32) -> bool {
        self.bool_segments.contains(&segment_id)
    }

    /// Record that `segment_id` stores a NumPy bool array.
    pub fn mark_bool_segment(&mut self, segment_id: u32) {
        self.bool_segments.insert(segment_id);
    }

    /// Record that the array `v` (static or runtime-shaped) is a NumPy bool
    /// array.
    pub fn mark_bool_array(&mut self, v: &Value) {
        match v {
            Value::StaticArray { segment_id, .. } => self.mark_bool_segment(*segment_id),
            Value::DynamicNDArray(d) => self.mark_bool_segment(d.segment_id),
            _ => {}
        }
    }

    /// Mark `segment_id`, freshly created from `cells`, as a bool segment
    /// when every cell is a bool cell (NumPy's `np.array([...])` of bools).
    pub fn note_segment_cells(&mut self, segment_id: u32, cells: &[Value]) {
        if !cells.is_empty() && cells.iter().all(|c| self.is_bool_cell(c)) {
            self.mark_bool_segment(segment_id);
        }
    }
}
