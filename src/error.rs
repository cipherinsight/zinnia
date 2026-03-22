use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ZinniaError {
    pub message: String,
}

impl fmt::Display for ZinniaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ZinniaError {}

impl From<ZinniaError> for PyErr {
    fn from(err: ZinniaError) -> PyErr {
        PyRuntimeError::new_err(err.message)
    }
}
