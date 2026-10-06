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
//!   holds a NumPy bool: a read from a bool segment (`ir_read_memory`), a
//!   select between two bool cells (`value_ops::select_value`), a `Bool`
//!   parameter element, a cell of a bool array's compile-time cache. The
//!   record is per statement, and an integer view of a `Boolean` statement
//!   (`ir_int_cast` keeps the statement) is never recorded: it is NumPy's
//!   integer of that bool (`astype(int)`), while the `Boolean` itself is a
//!   bool cell by its type;
//! * a **bool segment** stores a bool array; an array is a bool array when
//!   its segment is one. A segment becomes one when it is created from bool
//!   cells only (`segment::alloc_and_write_values`,
//!   `static_array::base::build_static_array_from_flat`), or when an
//!   operation whose NumPy result is a bool array marks it
//!   ([`IRBuilder::mark_bool_segment`]: a boolean mask applied to a bool
//!   array, a sort of one, `dtype=bool` constructors). A view shares its
//!   base's segment and so its dtype. Writes do not change a segment's dtype
//!   (NumPy converts the assigned value to the array's dtype:
//!   [`IRBuilder::bool_dtype_cell`] for a bool array).
//!
//! The cells of a bool array are bool cells however they are read: from
//! memory (`ir_read_memory` records them) or from the compile-time cache
//! (`static_array_payload`, kept in bool cells by
//! [`IRBuilder::mark_bool_segment`] and by the writes). A conversion to an
//! integer dtype makes integer cells ([`IRBuilder::int_dtype_cell`]).
//!
//! Recording a dtype emits no IR; it only decides how an array used as an
//! index is read.

use crate::types::{ScalarValue, Value};

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
    /// Only for values whose NumPy dtype is bool. An integer view of a
    /// `Boolean` statement is not recorded (it is an integer; see the module
    /// documentation).
    pub fn record_bool_cell(&mut self, v: &Value) {
        if let Value::Integer(ScalarValue { stmt_id: Some(sid), .. }) = v {
            if !self.yields_boolean(*sid) {
                self.bool_cells.insert(*sid);
            }
        }
    }

    /// Whether statement `sid` produces a `Value::Boolean`.
    fn yields_boolean(&self, sid: crate::types::StmtId) -> bool {
        self.stmts
            .get(sid as usize)
            .is_some_and(|stmt| matches!(super::unknown_value_for_ir(&stmt.ir, sid), Value::Boolean(_)))
    }

    /// The `Boolean` that the integer `v` is a view of (`ir_int_cast` of a
    /// `Boolean` keeps its statement), if it is one.
    fn boolean_of_view(&self, v: &Value) -> Option<Value> {
        let Value::Integer(s) = v else { return None };
        if !s.stmt_id.is_some_and(|sid| self.yields_boolean(sid)) {
            return None;
        }
        Some(Value::Boolean(ScalarValue {
            static_val: s.static_val.map(|n| n != 0),
            stmt_id: s.stmt_id,
            value_id: s.value_id,
        }))
    }

    /// Whether `segment_id` stores a NumPy bool array.
    pub fn is_bool_segment(&self, segment_id: u32) -> bool {
        self.bool_segments.contains(&segment_id)
    }

    /// Record that `segment_id` stores a NumPy bool array. Its cached cells
    /// (an operation whose NumPy result is bool, e.g. a sort of a bool
    /// array, may cache integer cells) become bool cells; this emits no IR,
    /// so it is only for cells whose NumPy dtype is bool.
    pub fn mark_bool_segment(&mut self, segment_id: u32) {
        self.bool_segments.insert(segment_id);
        let Some(cells) = self.static_array_payload.get(&segment_id).cloned() else {
            return;
        };
        let cells = cells
            .iter()
            .map(|cell| match self.boolean_of_view(cell) {
                Some(boolean) => boolean,
                None => {
                    self.record_bool_cell(cell);
                    cell.clone()
                }
            })
            .collect();
        self.static_array_payload.insert(segment_id, cells);
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

    /// `v` as a cell of a NumPy bool array (assigned into one, or converted
    /// with `dtype=bool`): its truth value, as a bool cell. A bool cell stays
    /// as it is, an integer view of a `Boolean` is that `Boolean`, and any
    /// other number `x` becomes `x != 0`.
    pub fn bool_dtype_cell(&mut self, v: &Value) -> Value {
        if self.is_bool_cell(v) {
            return v.clone();
        }
        if let Some(boolean) = self.boolean_of_view(v) {
            return boolean;
        }
        if let Some(n) = v.int_val() {
            return self.ir_constant_bool(n != 0);
        }
        self.ir_bool_cast(v)
    }

    /// `v` as a cell of a NumPy integer array (`astype(int)`, `int(v)`,
    /// `dtype=int`): the same number, not a bool cell, so an array of them
    /// indexes by position. A `Boolean` becomes its integer view (no IR); a
    /// recorded bool cell becomes the new statement `v + 0`, since the record
    /// is per statement. Other values are returned unchanged (a float is
    /// not converted here).
    pub fn int_dtype_cell(&mut self, v: &Value) -> Value {
        match v {
            Value::Boolean(_) => self.ir_int_cast(v),
            Value::Integer(_) if self.is_bool_cell(v) => {
                let zero = self.ir_constant_int(0);
                self.ir_add_i(v, &zero)
            }
            _ => v.clone(),
        }
    }
}
