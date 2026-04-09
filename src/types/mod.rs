pub mod zinnia_type;
pub mod scalar;
pub mod composite;
pub mod value;

#[cfg(test)]
mod tests;

/// Statement ID type used throughout the IR system.
pub type StmtId = u32;

// Re-export all public types.
pub use zinnia_type::{AnnotationArg, NumberType, ZinniaType};
pub use scalar::{ScalarValue, StringValue};
pub use composite::{CompositeData, DynArrayMeta, DynamicNDArrayData, NDArrayData};
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
