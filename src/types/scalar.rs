use super::StmtId;

/// Holds an optional compile-time constant value and an optional IR statement
/// pointer. Replaces the ValueTriplet pattern for atomic types.
#[derive(Debug, Clone)]
pub struct ScalarValue<T: Clone> {
    /// Compile-time constant value (None if unknown at compile time).
    pub static_val: Option<T>,
    /// IR statement ID that produces this value (None if pure constant).
    pub ptr: Option<StmtId>,
}

impl<T: Clone> ScalarValue<T> {
    pub fn new(static_val: Option<T>, ptr: Option<StmtId>) -> Self {
        Self { static_val, ptr }
    }

    pub fn constant(val: T) -> Self {
        Self {
            static_val: Some(val),
            ptr: Option::None,
        }
    }

    pub fn runtime(ptr: StmtId) -> Self {
        Self {
            static_val: Option::None,
            ptr: Some(ptr),
        }
    }

    pub fn known(val: T, ptr: StmtId) -> Self {
        Self {
            static_val: Some(val),
            ptr: Some(ptr),
        }
    }
}

impl<T: Clone + PartialEq> PartialEq for ScalarValue<T> {
    fn eq(&self, other: &Self) -> bool {
        self.static_val == other.static_val && self.ptr == other.ptr
    }
}

// ---------------------------------------------------------------------------
// StringValue
// ---------------------------------------------------------------------------

/// String value with compile-time known string and an IR statement reference.
#[derive(Debug, Clone)]
pub struct StringValue {
    pub val: String,
    pub ptr: StmtId,
}
