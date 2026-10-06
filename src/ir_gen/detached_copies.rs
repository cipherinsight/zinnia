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
//! records detached pairs and, after every write, marks every segment still
//! linked to the written one as *stale*. Reading a name whose array lives in
//! a stale segment is a compile-time error, so such a program is rejected
//! instead of proving a result NumPy would not compute.

use std::collections::{BTreeSet, HashMap};

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
            Link::View => {
                "it is linked by NumPy view semantics to an array that was modified afterwards, \
                 but Zinnia had to copy it (the view's position depends on a private index, or \
                 it is a reversed or runtime-bounded slice), so it would not show the \
                 modification. Index the original array directly instead of keeping the view."
            }
            Link::PathDependentAlias => {
                "a name that refers to this array on some paths and to another array on others \
                 (bound under a runtime condition) was used to modify one of them; which array \
                 changed depends on the path, and Zinnia does not track that. Write to the \
                 arrays under the condition instead (`if c: a[i] = v else: b[i] = v`)."
            }
        }
    }
}

/// Detached (copy, base) segment pairs and the segments made stale by later
/// writes.
#[derive(Debug, Default)]
pub(crate) struct DetachedCopies {
    pairs: Vec<(u32, u32, Link)>,
    stale: HashMap<u32, &'static str>,
}

impl DetachedCopies {
    /// Segments connected to `segment` through detached pairs (transitively),
    /// excluding `segment` itself, with the kind of the first link that
    /// reached each.
    fn linked(&self, segment: u32) -> Vec<(u32, Link)> {
        let mut seen = BTreeSet::from([segment]);
        let mut frontier = vec![segment];
        let mut found = Vec::new();
        while let Some(current) = frontier.pop() {
            for &(copy, base, link) in &self.pairs {
                let other = if copy == current {
                    base
                } else if base == current {
                    copy
                } else {
                    continue;
                };
                if seen.insert(other) {
                    found.push((other, link));
                    frontier.push(other);
                }
            }
        }
        found
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
            self.detached_copies.pairs.push((result_segment, base_segment, Link::View));
        }
    }

    /// Record `merged = select(c, a, b)` for a name bound to `a` on one path
    /// and to `b` on another: Python would make the name the same object as
    /// one of them, so a later write through the name (or to `a` or `b`)
    /// detaches the others.
    pub(crate) fn record_path_dependent_alias(&mut self, merged: &Value, sources: [&Value; 2]) {
        let Some(merged_segment) = array_segment(merged) else { return };
        let source_segments: Vec<u32> = sources.iter().filter_map(|v| array_segment(v)).collect();
        // A merge under a compile-time condition is one of the sources itself.
        if source_segments.contains(&merged_segment) {
            return;
        }
        for source_segment in source_segments {
            self.detached_copies
                .pairs
                .push((merged_segment, source_segment, Link::PathDependentAlias));
        }
    }

    /// `select(c, a, b)` for the value of a name after a branch, loop or
    /// conditional assignment, with the aliasing recorded (see
    /// [`Self::record_path_dependent_alias`]).
    pub(crate) fn merge_binding(&mut self, condition: &Value, a: &Value, b: &Value) -> Value {
        let merged = crate::helpers::value_ops::select_value(&mut self.builder, condition, a, b);
        self.record_path_dependent_alias(&merged, [a, b]);
        merged
    }

    /// Note a write into the memory of `written` (an array value): every
    /// array detached from it no longer matches NumPy.
    pub(crate) fn note_array_write(&mut self, written: &Value) {
        let Some(segment) = array_segment(written) else { return };
        for (other, link) in self.detached_copies.linked(segment) {
            self.detached_copies.stale.entry(other).or_insert(link.stale_reason());
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
