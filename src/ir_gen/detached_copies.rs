//! Copies that NumPy would treat as views.
//!
//! NumPy basic indexing (integers, slices, `...`, `None`) returns a *view*:
//! the result shares memory with the indexed array, so a write through either
//! one is visible through the other. Zinnia keeps that sharing when the
//! view's layout is a compile-time constant (same memory segment, static
//! offset and strides). When the layout depends on a private index, or uses
//! a reversed or runtime-bounded slice, Zinnia materialises a copy instead.
//!
//! The same happens when a name is bound to one of two arrays depending on a
//! runtime condition (`c = a` on one path, `c = b` on the other): Python makes
//! `c` the very same object as `a` or `b`, but Zinnia merges the two into a
//! new array that selects between them.
//!
//! A copy and the array it was taken from are *detached*: after a write to
//! either one, the other no longer holds what NumPy would show. This module
//! decides which pairs are detached and records them in the array store
//! (`IRBuilder::link_storages`). The store does the rest in one place each:
//! every write to a storage (any syntax: assignment, `op=`, `out=`, a
//! helper, a method) marks every storage linked to it *stale*, and every
//! read of a stale storage's cells (a name, a method call, an element of a
//! nested container, a merge's operand, an output) is a compile-time error,
//! so such a program is rejected instead of proving a result NumPy would not
//! compute. A merge of a stale value carries the staleness over to its
//! result (`helpers::value_ops::select_value`), so a merge that is never
//! read costs nothing.

use crate::types::{SliceIndex, Value};

use super::IRGenerator;

/// Why Python would share memory that Zinnia copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Link {
    /// The copy is a NumPy view of the base.
    View,
    /// The copy merges arrays a name was bound to on different paths.
    PathDependentAlias,
}

impl Link {
    fn stale_reason(self) -> &'static str {
        match self {
            Link::View => crate::builder::COPIED_VIEW,
            Link::PathDependentAlias => {
                "a name that refers to this array on some paths and to another array on others \
                 (bound under a runtime condition) was used to modify one of them; which array \
                 changed depends on the path, and Zinnia does not track that. Write to the \
                 arrays under the condition instead (`if c: a[i] = v else: b[i] = v`)."
            }
        }
    }
}

/// Why a name bound to arrays of different shapes cannot be used.
const DIFFERENT_SHAPES: &str = "the name is bound to arrays of different shapes (or to an array \
     and a number) on different paths (a runtime condition, or a loop that may stop early), and \
     Zinnia represents an array with one shape. Give it the same shape on every path, or use a \
     different name for each shape.";

/// The memory segments of the arrays in `value`: an array's own, or those of
/// the arrays a list or tuple holds at any depth (a merge of two arrays of
/// different shapes is a list of rows).
pub(crate) fn array_segments(value: &Value) -> Vec<u32> {
    match value {
        Value::List(data) | Value::Tuple(data) => data.values.iter().flat_map(array_segments).collect(),
        other => array_segment(other).into_iter().collect(),
    }
}

/// The memory segment holding an array value, if it has one.
pub(crate) fn array_segment(value: &Value) -> Option<u32> {
    match value {
        Value::StaticArray { segment_id, .. } => Some(*segment_id),
        Value::DynamicNDArray(data) => Some(data.segment_id),
        _ => None,
    }
}

/// Whether `indices` is NumPy basic indexing (the result is a view).
fn is_basic_indexing(indices: &[SliceIndex]) -> bool {
    indices.iter().all(|index| match index {
        SliceIndex::Single(value) => matches!(value, Value::Integer(_)),
        SliceIndex::Range(..) | SliceIndex::Ellipsis | SliceIndex::NewAxis => true,
    })
}

impl IRGenerator {
    /// Record `result = base[indices]` when NumPy returns a view but the
    /// result does not share `base`'s memory.
    pub(crate) fn record_subscript_copy(&mut self, base: &Value, indices: &[SliceIndex], result: &Value) {
        let (Some(base_segment), Some(result_segment)) = (array_segment(base), array_segment(result))
        else {
            return;
        };
        if base_segment != result_segment && is_basic_indexing(indices) {
            self.builder.link_storages(result_segment, base_segment, Link::View.stale_reason());
        }
    }

    /// Record `result = f(base, ...)` for a function or method that NumPy
    /// always answers with a view of `base` (`np.flip`, `np.rot90`, `.T`,
    /// ...), when Zinnia's result does not share `base`'s memory: a later
    /// write to either one makes the other stale (a compile error when it is
    /// read), instead of a result that silently misses the write.
    pub(crate) fn record_view_result(&mut self, base: &Value, result: &Value) {
        let (Some(base_segment), Some(result_segment)) = (array_segment(base), array_segment(result)) else {
            return;
        };
        if base_segment != result_segment {
            self.builder.link_storages(result_segment, base_segment, Link::View.stale_reason());
        }
    }

    /// `x` (an array) as `dtype` where NumPy copies only when it must
    /// (`np.asarray(x, dtype)`, `np.array(x, dtype, copy=None)`,
    /// `x.astype(dtype, copy=False)`; `dtype` `None`: no conversion): `x`
    /// itself when the dtype is `x`'s own, a converted copy otherwise. Zinnia
    /// knows an integer or bool array's NumPy dtype exactly, but not a float
    /// or complex array's width (float32 and float64 are both Q32), so for
    /// those it cannot tell whether NumPy returns `x`: it returns a copy
    /// linked to `x` as a view (a write through either one, then a read of
    /// the other, is a compile error). With `must_not_copy`
    /// (`np.array(x, copy=False)`), a conversion NumPy must copy for raises
    /// `ValueError`, as does one Zinnia cannot decide.
    pub(crate) fn as_dtype_without_copy(
        &mut self,
        x: &Value,
        dtype: Option<crate::helpers::dtype::Dtype>,
        must_not_copy: bool,
        context: &str,
    ) -> Value {
        use crate::helpers::dtype::Dtype;
        let Some(dtype) = dtype else { return x.clone() };
        let own = crate::helpers::dtype::element_dtype(&mut self.builder, x);
        let exact = matches!(own, Dtype::Bool | Dtype::Int64 | Dtype::FixedInt(_));
        if own == dtype && exact {
            return x.clone();
        }
        if must_not_copy {
            panic!(
                "{context}: copy=False needs the array's own dtype, and Zinnia cannot tell that \
                 {dtype:?} is (NumPy raises ValueError when a copy is needed); drop copy=False"
            );
        }
        let converted = crate::helpers::dtype::convert(&mut self.builder, x, dtype);
        if own == dtype {
            self.record_view_result(x, &converted);
        }
        converted
    }

    /// Record `merged = select(c, a, b)` for a name bound to `a` on one path
    /// and to `b` on another: Python would make the name the same object as
    /// one of them, so a later write through the name (or to `a` or `b`)
    /// detaches the others.
    pub(crate) fn record_path_dependent_alias(&mut self, condition: &Value, merged: &Value, sources: [&Value; 2]) {
        // A merge under a compile-time condition is one of the sources itself.
        if condition.bool_val().is_some() || condition.int_val().is_some() {
            return;
        }
        let [a, b] = sources.map(array_segments);
        let why = Link::PathDependentAlias.stale_reason();
        for merged_segment in array_segments(merged) {
            // The same array on both paths (a list element both sides hold).
            if a.contains(&merged_segment) && b.contains(&merged_segment) {
                continue;
            }
            // A merged array may reuse one side's storage; it still stands
            // for the other side's array where the condition selects it.
            for (segments, taken) in [(&a, true), (&b, false)] {
                for &source_segment in segments.iter().filter(|&&s| s != merged_segment) {
                    self.builder.link_merged_storage(merged_segment, source_segment, why, condition, taken);
                }
            }
        }
    }

    /// `select(c, a, b)` for the value of a name after a branch, loop or
    /// conditional assignment, with the aliasing recorded (see
    /// [`Self::record_path_dependent_alias`]).
    pub(crate) fn merge_binding(&mut self, condition: &Value, a: &Value, b: &Value) -> Value {
        if let Some(stand_in) = self.merge_of_different_shapes(condition, a, b) {
            return stand_in;
        }
        self.merge_values(condition, a, b)
    }

    /// [`Self::merge_binding`] for a value that is used directly, not bound
    /// to a name (a function's result merged over its `return`s): arrays of
    /// different shapes are a compile error at the merge, since such a value
    /// reaches code (a loop over it, its `.shape` in an expression) that
    /// reads its shape without a name lookup to check it.
    pub(crate) fn merge_values(&mut self, condition: &Value, a: &Value, b: &Value) -> Value {
        let (a_cells, b_cells) = self.static_side_as_cells(a, b);
        let merged =
            crate::helpers::value_ops::select_value(&mut self.builder, condition, &a_cells, &b_cells);
        self.record_path_dependent_alias(condition, &merged, [a, b]);
        merged
    }

    /// A name bound to arrays of different shapes on different paths, which
    /// no single array represents (multi-dimensional arrays of different
    /// shapes, or an array and a number; 1-D arrays of different lengths
    /// merge into a runtime-shaped array): a stand-in array whose cells and
    /// shape cannot be read or written, so the program is rejected only if
    /// it uses the name before rebinding it. `None` when the two merge.
    fn merge_of_different_shapes(&mut self, condition: &Value, a: &Value, b: &Value) -> Option<Value> {
        if condition.bool_val().is_some() || condition.int_val().is_some() {
            return None; // `select_value` takes one side
        }
        let is_number = |v: &Value| {
            matches!(v, Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. })
        };
        let array = match (a, b) {
            (Value::StaticArray { shape: sa, .. }, Value::StaticArray { shape: sb, .. })
                if sa != sb && (sa.len() > 1 || sb.len() > 1) => a,
            (Value::StaticArray { .. }, other) if is_number(other) => a,
            (other, Value::StaticArray { .. }) if is_number(other) => b,
            _ => return None,
        };
        // An empty array has no storage of its own to stand for the merge.
        if matches!(array, Value::StaticArray { shape, .. } if shape.contains(&0)) {
            panic!("Binding a name here is not supported: {DIFFERENT_SHAPES}");
        }
        let stand_in = self.builder.with_stale_reads(|builder| {
            let cells = crate::helpers::static_array::base::deep_to_value_list(builder, array);
            crate::helpers::static_array::base::to_static_array(builder, &cells)
                .expect("the cells of an array make an array")
        });
        // The stand-in is identified by its storage, so it needs one of its
        // own (an empty array's storage may be shared).
        let segment = array_segment(&stand_in).expect("a static array has a storage");
        let shared = self.different_shape_merges.contains_key(&segment)
            || array_segments(a).contains(&segment)
            || array_segments(b).contains(&segment);
        if shared {
            panic!("Binding a name here is not supported: {DIFFERENT_SHAPES}");
        }
        self.builder.mark_shape_unknown_in(&stand_in, DIFFERENT_SHAPES);
        self.different_shape_merges.insert(segment, (condition.clone(), a.clone(), b.clone()));
        Some(stand_in)
    }

    /// The array a name bound to `value` holds where the current code runs:
    /// `value` itself, or for a stand-in of arrays of different shapes the
    /// merged array the current path selects, when its condition is decided
    /// there (a read in a later iteration of the loop that rebinds the name,
    /// under the condition that the loop went on). Otherwise the stand-in.
    pub(crate) fn resolve_different_shapes(&mut self, value: Value) -> Value {
        let Some(segment) = array_segment(&value) else { return value };
        let Some((condition, a, b)) = self.different_shape_merges.get(&segment).cloned() else {
            return value;
        };
        match self.builder.decided_here(&condition) {
            Some(true) => self.resolve_different_shapes(a),
            Some(false) => self.resolve_different_shapes(b),
            None => value,
        }
    }

    /// When one side of a merge has a runtime shape and the other is a
    /// static array, `select_value` promotes the static side to a runtime-
    /// shaped array from its element values; hand it the static array's
    /// cells as a (nested) list, which is what the promotion reads.
    fn static_side_as_cells(&mut self, a: &Value, b: &Value) -> (Value, Value) {
        let as_cells = |builder: &mut crate::builder::IRBuilder, v: &Value| match v {
            Value::StaticArray { .. } => crate::helpers::static_array::base::to_value_list(builder, v),
            other => other.clone(),
        };
        match (a, b) {
            (Value::DynamicNDArray(_), Value::StaticArray { .. })
            | (Value::StaticArray { .. }, Value::DynamicNDArray(_)) => {
                (as_cells(&mut self.builder, a), as_cells(&mut self.builder, b))
            }
            _ => (a.clone(), b.clone()),
        }
    }

    /// Note an update of the array `written` (or of the arrays a list or
    /// tuple holds) whose new contents may live in another storage (a
    /// rebinding update): its links become stale as after a store write.
    /// (Writes into a storage are noted by the store itself.)
    pub(crate) fn note_array_write(&mut self, written: &Value) {
        for segment in array_segments(written) {
            self.builder.note_storage_replaced(segment);
        }
    }

    /// Note that the arrays of `old` now live in those of `new` (an update
    /// that rebinds a name to new storages): the new storages take over the
    /// old ones' links. Pairwise when the two hold as many arrays, otherwise
    /// every new one takes over every old one's links.
    pub(crate) fn note_array_moved(&mut self, old: &Value, new: &Value) {
        let (old, new) = (array_segments(old), array_segments(new));
        if old.len() == new.len() {
            for (o, n) in old.into_iter().zip(new) {
                self.builder.move_links(o, n);
            }
        } else {
            for &o in &old {
                for &n in &new {
                    self.builder.move_links(o, n);
                }
            }
        }
    }

    /// Reject reading `name` when its value (or an array at any depth inside
    /// it) lives in a stale storage. The store rejects the read of the cells
    /// anyway; this check names the variable in the error.
    pub(crate) fn check_not_stale(&mut self, name: &str, value: &Value) {
        if let Some(reason) = self.builder.stale_reason_in(value) {
            panic!("Reading `{}` is not supported here: {}", name, reason);
        }
    }
}
