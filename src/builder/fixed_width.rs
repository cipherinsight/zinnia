//! Values of NumPy's fixed-width integer types (`np.uint8`, `np.int16`, ...).
//!
//! A conversion into such a type (`helpers::dtype`) produces NumPy's value:
//! an int64 in the type's range. What NumPy does with that value next depends
//! on its type, which Zinnia's values do not carry:
//!
//! * arithmetic: NumPy computes `uint8 + uint8` modulo `2^8`, and mixes types
//!   by its promotion rules; Zinnia computes in int64;
//! * assignment into an array of the type: NumPy converts the assigned value
//!   into the type; Zinnia stores it as it is.
//!
//! Zinnia implements neither, so it rejects both. [`FixedWidthValues`]
//! records which IR statements hold a fixed-width value and which memory
//! segments hold such values, and [`IRBuilder::create_ir`] consults it for
//! every statement it builds (`track_statement`):
//!
//! * integer arithmetic with a tracked operand is a compile error (with a
//!   float operand it computes a float, as NumPy does, and is allowed);
//! * a statement whose result depends only on its operands' values — a
//!   comparison, a cast to float or bool, a check, an output — is allowed and
//!   untracked (those values equal NumPy's);
//! * a selection (`np.where`, `max`, a branch merge, a sorted value) is
//!   tracked as the type of its candidates, or as `Derived` when they differ;
//! * a memory read is tracked as what its segment holds, and a write of an
//!   untracked value (or one of another type) into a segment that holds
//!   fixed-width values is a compile error. Static arrays whose cells live
//!   only in the compile-time payload register their content when built
//!   (`note_static_array_cells`), and subscript assignments are checked
//!   before they are lowered (`check_assignment`), because some write paths
//!   update the payload without a memory write.
//!
//! The classification is an exhaustive match over the IR, so a new operation
//! has to be classified. Lowerings that select or index with arithmetic
//! (boolean-mask compaction, an index used in an address) are rejected like
//! user arithmetic: convert first. A conversion back to int64
//! (`astype(np.int64)`, `int(x)`) is a fresh untracked copy
//! ([`IRBuilder::untracked_int64`]).
//!
//! Tracking starts at the first conversion into a fixed-width type, so a
//! program without one compiles exactly as before.

use std::collections::HashMap;
use std::ops::Range;

use crate::helpers::dtype::IntType;
use crate::ir_defs::{HintKind, IR};
use crate::types::{StmtId, Value};

use super::IRBuilder;

/// What a tracked statement, or a tracked segment's cells, hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixedWidth {
    /// Values of one fixed-width type.
    Exactly(IntType),
    /// Values computed from fixed-width values whose NumPy type the compiler
    /// does not know (several types, or fixed-width and other values mixed).
    Derived,
}

impl FixedWidth {
    /// The type of a value that is one of `a` and `b` (`None`: untracked).
    fn either(a: Option<FixedWidth>, b: Option<FixedWidth>) -> Option<FixedWidth> {
        match (a, b) {
            (None, None) => None,
            (Some(x), Some(y)) if x == y => Some(x),
            _ => Some(FixedWidth::Derived),
        }
    }
}

/// The cells of a memory segment allocated while tracking is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Content {
    /// Allocated with fill value 0 (which every type holds), not written.
    Fresh,
    /// Untracked values only.
    Untracked,
    Tracked(FixedWidth),
}

/// The tracking state of one compilation (see the module doc).
#[derive(Default)]
pub(crate) struct FixedWidthValues {
    active: bool,
    /// Set while the conversion helpers copy a tracked value.
    suspended: bool,
    values: HashMap<StmtId, FixedWidth>,
    /// Segments allocated before tracking began are absent and hold
    /// untracked values.
    segments: HashMap<u32, Content>,
}

impl FixedWidthValues {
    fn of(&self, v: &Value) -> Option<FixedWidth> {
        if !self.active {
            return None;
        }
        self.values.get(&v.stmt_id()?).copied()
    }

    fn content(&self, segment: u32) -> Content {
        self.segments
            .get(&segment)
            .copied()
            .unwrap_or(Content::Untracked)
    }
}

/// How a statement's result relates to its operands' types.
enum Role {
    /// Integer or float arithmetic: NumPy computes it in the operands' type.
    Arithmetic,
    /// No operands, or a result that depends only on the operands' values.
    ValueOnly,
    /// The result is one of the operands in the range.
    Selects(Range<usize>),
    /// Allocates a segment filled with the value.
    Allocates(u32, i64),
    Reads(u32),
    /// Writes its second operand.
    Writes(u32),
    /// Passes the values to an external function, which computes on them.
    External,
}

fn role(ir: &IR, operands: usize) -> Role {
    match ir {
        IR::AddI
        | IR::SubI
        | IR::MulI
        | IR::DivI
        | IR::FloorDivI
        | IR::FloorDivIPinned { .. }
        | IR::ModI
        | IR::PowI
        | IR::AbsI
        | IR::InvI
        | IR::BitAndI
        | IR::BitOrI
        | IR::BitXorI
        | IR::ShlI
        | IR::ShrI
        | IR::BitNotI => Role::Arithmetic,
        IR::AddF
        | IR::SumF { .. }
        | IR::SubF
        | IR::MulF
        | IR::ProductSumF { .. }
        | IR::DivF
        | IR::FloorDivF
        | IR::ModF
        | IR::PowF
        | IR::PowFI
        | IR::AbsF
        | IR::SignF
        | IR::SinF
        | IR::SinHF
        | IR::CosF
        | IR::CosHF
        | IR::TanF
        | IR::TanHF
        | IR::SqrtF
        | IR::ExpF
        | IR::LogF
        | IR::ArcSinF
        | IR::ArcCosF
        | IR::ArcTan2F
        | IR::AddStr => Role::Arithmetic,
        IR::SignI => Role::Selects(0..operands),
        IR::SelectI
        | IR::SelectF
        | IR::SelectB
        | IR::SelectBitI
        | IR::SelectBitF
        | IR::SelectManyI { .. }
        | IR::SelectManyF { .. } => Role::Selects(1..operands),
        IR::Hint {
            kind: HintKind::StableArgsort,
            ..
        } => Role::ValueOnly,
        IR::Hint {
            kind: HintKind::SortedValues | HintKind::MaskedSortedValues,
            len,
        } => Role::Selects(0..*len as usize),
        IR::Sorted { index: true, .. } => Role::ValueOnly,
        IR::Sorted {
            index: false, len, ..
        } => Role::Selects(0..*len as usize),
        IR::AllocateMemory {
            segment_id,
            init_value,
            ..
        } => Role::Allocates(*segment_id, *init_value),
        IR::ReadMemory { segment_id } | IR::DynamicNDArrayGetItem { segment_id, .. } => {
            Role::Reads(*segment_id)
        }
        IR::WriteMemory { segment_id } | IR::DynamicNDArraySetItem { segment_id, .. } => {
            Role::Writes(*segment_id)
        }
        IR::InvokeExternal { .. } | IR::ExportExternalI { .. } | IR::ExportExternalF { .. } => {
            Role::External
        }
        IR::ConstantInt { .. }
        | IR::ConstantFloat { .. }
        | IR::ConstantBool { .. }
        | IR::ConstantStr { .. }
        | IR::EqI
        | IR::NeI
        | IR::LtIBounded
        | IR::GtIBounded
        | IR::AssertU128Lt
        | IR::AssertU128LtBit { .. }
        | IR::IndexInBox { .. }
        | IR::LtI
        | IR::LteI
        | IR::GtI
        | IR::GteI
        | IR::EqF
        | IR::NeF
        | IR::LtF
        | IR::LteF
        | IR::GtF
        | IR::GteF
        | IR::KeyLess { .. }
        | IR::PackKey { .. }
        | IR::PackedKeyLess { .. }
        | IR::PackedKeyIndex { .. }
        | IR::LogicalAnd
        | IR::LogicalOr
        | IR::LogicalNot
        | IR::IntCast
        | IR::FloatCast
        | IR::BoolCast
        | IR::StrI
        | IR::StrF
        | IR::Print
        | IR::ReadInteger { .. }
        | IR::ReadFloat { .. }
        | IR::ReadHash { .. }
        | IR::ReadExternalResult { .. }
        | IR::Assert
        | IR::AssertBit
        | IR::AssertNonNegI
        | IR::ExposePublicI
        | IR::ExposePublicF
        | IR::StructuralPredicate { .. }
        | IR::ScalarPrecondition { .. }
        | IR::MemoryTraceEmit { .. }
        | IR::MemoryTraceSeal
        | IR::AllocateDynamicNDArrayMeta { .. }
        | IR::WitnessDynamicNDArrayMeta { .. }
        | IR::AssertDynamicNDArrayMeta { .. }
        | IR::PoseidonHash
        | IR::EqHash => Role::ValueOnly,
    }
}

fn describe(t: FixedWidth) -> String {
    match t {
        FixedWidth::Exactly(t) => format!("NumPy type {}", t.name()),
        FixedWidth::Derived => "a NumPy fixed-width integer type".to_string(),
    }
}

fn reject_arithmetic(ir: &IR, t: FixedWidth) -> ! {
    let modulus = match t {
        FixedWidth::Exactly(t) => format!("modulo 2^{}", t.bits),
        FixedWidth::Derived => "in the operands' types".to_string(),
    };
    panic!(
        "`{}` on a value of {}: NumPy computes it {modulus}, Zinnia only in int64; convert the \
         value first with .astype(np.int64) or int(...)",
        ir.signature(),
        describe(t)
    )
}

fn reject_assignment(t: FixedWidth) -> ! {
    panic!(
        "a value of another type is stored into an array of {} (an assignment, or an array \
         built from values of several types): NumPy converts it into the array's type, Zinnia \
         does not; store values of that type, or convert the values with .astype(...) first",
        describe(t)
    )
}

impl IRBuilder {
    /// Record or check statement `stmt` (see the module doc). Called by
    /// `create_ir` for every statement once tracking is active.
    pub(super) fn track_statement(&mut self, ir: &IR, stmt: StmtId, operands: &[Value]) {
        let state = &self.fixed_width;
        if !state.active {
            return;
        }
        match role(ir, operands.len()) {
            Role::ValueOnly => {}
            // With a float operand the statement computes a float: NumPy
            // converts the fixed-width value to float64 exactly (`k * x` for
            // an integer `k` is a `MulI` with the float `x`).
            Role::Arithmetic if operands.iter().any(|v| matches!(v, Value::Float(_))) => {}
            Role::Arithmetic | Role::External => {
                if let Some(t) = operands.iter().find_map(|v| state.of(v)) {
                    if !state.suspended {
                        reject_arithmetic(ir, t);
                    }
                }
            }
            Role::Selects(candidates) => {
                let mut tracked = operands[candidates.clone()].iter().map(|v| state.of(v));
                let first = tracked.next().flatten();
                let joined = tracked.fold(first, FixedWidth::either);
                if let Some(t) = joined {
                    self.fixed_width.values.insert(stmt, t);
                }
            }
            Role::Allocates(segment, fill) => {
                let content = if fill == 0 {
                    Content::Fresh
                } else {
                    Content::Untracked
                };
                self.fixed_width.segments.insert(segment, content);
            }
            Role::Reads(segment) => {
                if let Content::Tracked(t) = state.content(segment) {
                    self.fixed_width.values.insert(stmt, t);
                }
            }
            Role::Writes(segment) => {
                let written = state.of(&operands[1]);
                let content = after_write(state.content(segment), written, state.suspended);
                self.fixed_width.segments.insert(segment, content);
            }
        }
    }

    /// Track `v` as a value of the fixed-width type `t` (and start tracking).
    pub(crate) fn track_fixed_width(&mut self, v: &Value, t: IntType) {
        let stmt = v
            .stmt_id()
            .expect("a converted fixed-width value is an IR statement");
        self.fixed_width.active = true;
        self.fixed_width.values.insert(stmt, FixedWidth::Exactly(t));
    }

    /// Track the elements of the array `x` (whose buffer holds values of the
    /// fixed-width type `t`) as that type.
    pub(crate) fn track_fixed_width_array(&mut self, x: &Value, t: IntType) {
        let segment = match x {
            Value::StaticArray { segment_id, .. } => *segment_id,
            Value::DynamicNDArray(data) => data.segment_id,
            other => unreachable!("track_fixed_width_array of a {}", other.zinnia_type()),
        };
        self.fixed_width.active = true;
        self.fixed_width
            .segments
            .insert(segment, Content::Tracked(FixedWidth::Exactly(t)));
    }

    /// A new statement equal to the integer `v`, untracked: `v + 0`, which
    /// the optimizer folds away.
    pub(crate) fn fresh_copy(&mut self, v: &Value) -> Value {
        let zero = self.ir_constant_int(0);
        let suspended = std::mem::replace(&mut self.fixed_width.suspended, true);
        let copy = self.ir_add_i(v, &zero);
        self.fixed_width.suspended = suspended;
        copy
    }

    /// The integer `v` as a plain int64 value: `v` itself, or an untracked
    /// copy of a tracked fixed-width value.
    pub(crate) fn untracked_int64(&mut self, v: &Value) -> Value {
        if self.fixed_width.of(v).is_some() {
            self.fresh_copy(v)
        } else {
            v.clone()
        }
    }

    /// Register the cells of a static array built over `segment` from a
    /// compile-time payload (no memory write records them).
    pub(crate) fn note_static_array_cells(&mut self, segment: u32, cells: &[Value]) {
        if !self.fixed_width.active {
            return;
        }
        let content = cells.iter().fold(Content::Fresh, |content, cell| {
            after_write(content, self.fixed_width.of(cell), true)
        });
        self.fixed_width.segments.insert(segment, content);
    }

    /// The fixed-width type of the elements of `x` (a scalar, an array or a
    /// list), when the compiler tracks them as one. A compile error when they
    /// are `Derived` (their NumPy type is unknown).
    pub(crate) fn fixed_width_content(&self, x: &Value) -> Option<IntType> {
        match self.tracked_content(x)? {
            FixedWidth::Exactly(t) => Some(t),
            FixedWidth::Derived => panic!(
                "the NumPy type of values computed from fixed-width integers is not tracked; \
                 convert them with .astype(...) first"
            ),
        }
    }

    /// Whether `x` (a scalar, an array or a list) holds tracked values.
    pub(crate) fn fixed_width_content_of(&self, x: &Value) -> Option<FixedWidth> {
        self.tracked_content(x)
    }

    /// Reject `target[...] = value` when `target` holds fixed-width values
    /// and `value` is not of the same type.
    pub(crate) fn check_assignment(&self, target: &Value, value: &Value) {
        let Some(target_type) = self.tracked_content(target) else {
            return;
        };
        let value_type = self.tracked_content(value);
        let allowed = match target_type {
            FixedWidth::Exactly(_) => value_type == Some(target_type),
            FixedWidth::Derived => value_type.is_some(),
        };
        if !allowed {
            reject_assignment(target_type);
        }
    }

    /// What `x`'s elements hold: a scalar's statement, an array's segment, a
    /// list's elements.
    fn tracked_content(&self, x: &Value) -> Option<FixedWidth> {
        let state = &self.fixed_width;
        if !state.active {
            return None;
        }
        let segment_content = |segment: u32| match state.content(segment) {
            Content::Tracked(t) => Some(t),
            Content::Fresh | Content::Untracked => None,
        };
        match x {
            Value::StaticArray { segment_id, .. } => segment_content(*segment_id),
            Value::DynamicNDArray(data) => segment_content(data.segment_id),
            Value::List(data) | Value::Tuple(data) => {
                let mut leaves = data.values.iter().map(|v| self.tracked_content(v));
                let first = leaves.next().flatten();
                leaves.fold(first, FixedWidth::either)
            }
            scalar => state.of(scalar),
        }
    }
}

/// A segment's content after a write of a value of type `written`. Writing an
/// untracked value, or one of another type, into a segment of fixed-width
/// values is a compile error unless `unchecked`. A fixed-width value stored
/// among untracked (int64) values is an int64 value there, as NumPy converts
/// it into the array's type, which holds it exactly.
fn after_write(content: Content, written: Option<FixedWidth>, unchecked: bool) -> Content {
    match (content, written) {
        (Content::Fresh, None) => Content::Untracked,
        (Content::Fresh, Some(t)) => Content::Tracked(t),
        (Content::Untracked, _) => Content::Untracked,
        (Content::Tracked(t), Some(w)) if t == w || t == FixedWidth::Derived => Content::Tracked(t),
        (Content::Tracked(t), _) => {
            if !unchecked {
                reject_assignment(t);
            }
            Content::Tracked(FixedWidth::Derived)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UINT8: IntType = IntType { bits: 8, signed: false };

    fn tracked_constant(b: &mut IRBuilder, value: i64) -> Value {
        let v = b.ir_constant_int(value);
        b.track_fixed_width(&v, UINT8);
        v
    }

    #[test]
    #[should_panic(expected = "NumPy computes it modulo 2^8")]
    fn arithmetic_on_a_fixed_width_value_is_rejected() {
        let mut b = IRBuilder::new();
        let v = tracked_constant(&mut b, 200);
        let one = b.ir_constant_int(1);
        b.ir_add_i(&v, &one);
    }

    #[test]
    fn comparisons_selections_and_int64_copies_are_allowed() {
        let mut b = IRBuilder::new();
        let v = tracked_constant(&mut b, 200);
        let w = tracked_constant(&mut b, 3);
        let less = b.ir_less_than_i(&v, &w);
        let selected = b.ir_select_i(&less, &v, &w);
        assert_eq!(b.fixed_width.of(&selected), Some(FixedWidth::Exactly(UINT8)));
        assert_eq!(b.fixed_width.of(&less), None);
        let plain = b.untracked_int64(&selected);
        let one = b.ir_constant_int(1);
        b.ir_add_i(&plain, &one);
    }

    #[test]
    fn memory_holds_the_type_of_what_was_written() {
        let mut b = IRBuilder::new();
        let v = tracked_constant(&mut b, 200);
        let segment = b.alloc_segment_id();
        b.ir_allocate_memory(segment, 2, 0);
        let address = b.ir_constant_int(0);
        b.ir_write_memory(segment, &address, &v);
        let read = b.ir_read_memory(segment, &address);
        assert_eq!(b.fixed_width.of(&read), Some(FixedWidth::Exactly(UINT8)));
    }

    #[test]
    #[should_panic(expected = "stored into an array of NumPy type uint8")]
    fn storing_an_int64_value_into_fixed_width_memory_is_rejected() {
        let mut b = IRBuilder::new();
        let v = tracked_constant(&mut b, 200);
        let segment = b.alloc_segment_id();
        b.ir_allocate_memory(segment, 2, 0);
        let (first, second) = (b.ir_constant_int(0), b.ir_constant_int(1));
        b.ir_write_memory(segment, &first, &v);
        let plain = b.ir_constant_int(300);
        b.ir_write_memory(segment, &second, &plain);
    }

    #[test]
    fn nothing_is_tracked_before_a_conversion() {
        let mut b = IRBuilder::new();
        let v = b.ir_constant_int(200);
        let one = b.ir_constant_int(1);
        b.ir_add_i(&v, &one);
        assert!(!b.fixed_width.active);
        assert!(b.fixed_width.segments.is_empty());
    }
}
