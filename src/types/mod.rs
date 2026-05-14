pub mod zinnia_type;
pub mod scalar;
pub mod composite;
pub mod value;
pub mod envelope;

#[cfg(test)]
mod tests;

/// Statement ID type used throughout the IR system.
pub type StmtId = u32;

/// Compilation-layer identity of a `Value` (compiler.value-id-and-fact-leaves).
///
/// Distinct from [`StmtId`] (which identifies an IR statement). Every
/// `ScalarValue` and `DynamicNDArrayData` carries a `ValueId` minted at
/// construction; cloning a Value preserves its ValueId (the clone is
/// "the same Value"). Equality on Values ignores `ValueId` — two `Integer
/// 5` constants are equal as Values even if minted at different times.
///
/// `ContractTerm` leaves reference Values via `ContractVar::Value(ValueId)`;
/// the witness emitter is the one place that resolves ValueId back to
/// StmtId via `IRBuilder::value_id_to_stmt_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct ValueId(pub u64);

impl ValueId {
    /// Mint a fresh, globally-unique `ValueId`. Uses an `AtomicU64`
    /// counter — independent of any `IRBuilder`, so constructors that
    /// don't have one in scope can still mint.
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        ValueId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// Re-export all public types. (`ValueId` is defined in this module above.)
pub use zinnia_type::{AnnotationArg, NumberType, ZinniaType};
pub use scalar::{ScalarValue, StringValue};
pub use composite::{CompositeData, DynArrayMeta, DynamicNDArrayData, NDArrayData};
pub use envelope::{broadcast_envelopes, Dim, DimTable, DimVar, Envelope};
pub use value::Value;

/// Represents a slice index used inside an `array[…]` subscript.
///
/// `Single` and `Range` are the standard NumPy/Python slice forms.
/// `NewAxis` is `np.newaxis` / `None` — inserts a unit-length axis at this
/// position without consuming a source dimension.
/// `Ellipsis` is `...` — expands to as many full-range slices as needed to
/// align the remaining indices with the source rank.
#[derive(Clone)]
pub enum SliceIndex {
    Single(Value),
    Range(Option<Value>, Option<Value>, Option<Value>),
    NewAxis,
    Ellipsis,
}
