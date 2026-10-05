//! Copies that NumPy would treat as views.
//!
//! NumPy basic indexing (integers, slices, `...`, `None`) returns a *view*:
//! the result shares memory with the indexed array, so a write through either
//! one is visible through the other. Zinnia keeps that sharing when the
//! view's layout is a compile-time constant (same memory segment, static
//! offset and strides). When the layout depends on a private index, or uses
//! a reversed or runtime-bounded slice, Zinnia materialises a copy instead.
//!
//! A copy and the array it was taken from are *detached*: after a write to
//! either one, the other no longer holds what NumPy would show. This module
//! records detached pairs and, after every write, marks every segment still
//! linked to the written one as *stale*. Reading a name whose array lives in
//! a stale segment is a compile-time error, so such a program is rejected
//! instead of proving a result NumPy would not compute.

use std::collections::{BTreeSet, HashMap};

use crate::types::{SliceIndex, Value};

use super::IRGenerator;

/// Detached (copy, base) segment pairs and the segments made stale by later
/// writes.
#[derive(Debug, Default)]
pub(crate) struct DetachedCopies {
    pairs: Vec<(u32, u32)>,
    stale: HashMap<u32, String>,
}

impl DetachedCopies {
    /// Segments connected to `segment` through detached pairs (transitively),
    /// excluding `segment` itself.
    fn linked(&self, segment: u32) -> BTreeSet<u32> {
        let mut seen = BTreeSet::from([segment]);
        let mut frontier = vec![segment];
        while let Some(current) = frontier.pop() {
            for &(copy, base) in &self.pairs {
                let other = if copy == current {
                    base
                } else if base == current {
                    copy
                } else {
                    continue;
                };
                if seen.insert(other) {
                    frontier.push(other);
                }
            }
        }
        seen.remove(&segment);
        seen
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
            self.detached_copies.pairs.push((result_segment, base_segment));
        }
    }

    /// Note a write into the memory of `written` (an array value): every
    /// array detached from it no longer matches NumPy.
    pub(crate) fn note_array_write(&mut self, written: &Value) {
        let Some(segment) = array_segment(written) else { return };
        for other in self.detached_copies.linked(segment) {
            self.detached_copies.stale.entry(other).or_insert_with(|| {
                "it is linked by NumPy view semantics to an array that was modified afterwards, \
                 but Zinnia had to copy it (the view's position depends on a private index, or \
                 it is a reversed or runtime-bounded slice), so it would not show the \
                 modification. Index the original array directly instead of keeping the view."
                    .to_string()
            });
        }
    }

    /// Reject reading `name` when its value (or an array inside it) lives in
    /// a stale segment.
    pub(crate) fn check_not_stale(&self, name: &str, value: &Value) {
        let stale_reason = |v: &Value| array_segment(v).and_then(|s| self.detached_copies.stale.get(&s));
        let reason = match value {
            Value::List(data) | Value::Tuple(data) => data.values.iter().find_map(stale_reason),
            other => stale_reason(other),
        };
        if let Some(reason) = reason {
            panic!("Reading `{}` is not supported here: {}", name, reason);
        }
    }
}
