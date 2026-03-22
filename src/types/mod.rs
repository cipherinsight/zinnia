pub mod zinnia_type;
pub mod scalar;
pub mod composite;
pub mod value;
pub mod dt_serde;

#[cfg(test)]
mod tests;

/// Statement ID type used throughout the IR system.
pub type StmtId = u32;

// Re-export all public types for backward compatibility
pub use zinnia_type::{AnnotationArg, NumberType, ZinniaType};
pub use scalar::{ScalarValue, StringValue};
pub use composite::{CompositeData, DynArrayMeta, DynamicNDArrayData, NDArrayData};
pub use value::Value;
pub use dt_serde::DTDescriptorDict;

/// Represents a slice index: either a single value or a range (start, stop, step).
pub enum SliceIndex {
    Single(Value),
    Range(Option<Value>, Option<Value>, Option<Value>),
}
