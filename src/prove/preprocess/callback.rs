//! Callback trait for invoking external functions during preprocessing.

use crate::prove::error::ProvingError;
use crate::prove::types::Value;

/// Trait for invoking external functions during preprocessing.
///
/// Implementations:
/// - `PyExternalCallback`: calls Python functions via PyO3
/// - `FnExternalCallback`: pure-Rust closures for testing
pub trait ExternalCallback {
    /// Call an external function by name with the given arguments.
    /// Arguments and return value use the unified `Value` type.
    fn call(
        &self,
        func_name: &str,
        args: Vec<Value>,
    ) -> Result<Value, ProvingError>;
}

/// A no-op callback that rejects all external calls.
pub struct NoExternalCallback;

impl ExternalCallback for NoExternalCallback {
    fn call(&self, func_name: &str, _args: Vec<Value>) -> Result<Value, ProvingError> {
        Err(ProvingError::other(format!(
            "External function '{}' called but no callback registered",
            func_name
        )))
    }
}

/// A callback backed by a closure (for testing).
pub struct FnExternalCallback<F: Fn(&str, Vec<Value>) -> Result<Value, ProvingError>> {
    pub func: F,
}

impl<F> ExternalCallback for FnExternalCallback<F>
where
    F: Fn(&str, Vec<Value>) -> Result<Value, ProvingError>,
{
    fn call(&self, func_name: &str, args: Vec<Value>) -> Result<Value, ProvingError> {
        (self.func)(func_name, args)
    }
}
