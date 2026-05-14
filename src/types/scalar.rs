use super::{StmtId, ValueId};

/// Holds an optional compile-time constant value and an optional IR statement
/// reference. Replaces the ValueTriplet pattern for atomic types.
///
/// Carries a `value_id` minted at construction
/// (compiler.value-id-and-fact-leaves). Cloning preserves the
/// `value_id` — the clone is "the same Value." `PartialEq` ignores
/// `value_id` to preserve the existing semantic equality (`Integer 5`
/// is `Integer 5` regardless of mint history).
#[derive(Debug, Clone)]
pub struct ScalarValue<T: Clone> {
    /// Compile-time constant value (None if unknown at compile time).
    pub static_val: Option<T>,
    /// IR statement ID that produces this value (None if pure constant).
    pub stmt_id: Option<StmtId>,
    /// Compilation-layer identity. Minted fresh at every construction
    /// site (via `ValueId::next()`); preserved across `Clone`.
    pub value_id: ValueId,
}

impl<T: Clone> ScalarValue<T> {
    pub fn new(static_val: Option<T>, stmt_id: Option<StmtId>) -> Self {
        Self {
            static_val,
            stmt_id,
            value_id: ValueId::next(),
        }
    }

    pub fn constant(val: T) -> Self {
        Self {
            static_val: Some(val),
            stmt_id: Option::None,
            value_id: ValueId::next(),
        }
    }

    pub fn runtime(stmt_id: StmtId) -> Self {
        Self {
            static_val: Option::None,
            stmt_id: Some(stmt_id),
            value_id: ValueId::next(),
        }
    }

    pub fn known(val: T, stmt_id: StmtId) -> Self {
        Self {
            static_val: Some(val),
            stmt_id: Some(stmt_id),
            value_id: ValueId::next(),
        }
    }
}

impl<T: Clone + PartialEq> PartialEq for ScalarValue<T> {
    fn eq(&self, other: &Self) -> bool {
        // `value_id` deliberately excluded: two `Integer 5` constants with
        // distinct mint histories are equal as Values.
        self.static_val == other.static_val && self.stmt_id == other.stmt_id
    }
}

// ---------------------------------------------------------------------------
// StringValue
// ---------------------------------------------------------------------------

/// String value with compile-time known string and an IR statement reference.
#[derive(Debug, Clone)]
pub struct StringValue {
    pub val: String,
    pub stmt_id: StmtId,
}
