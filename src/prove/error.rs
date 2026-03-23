use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::fmt;

/// Errors specific to the proving pipeline.
#[derive(Debug, Clone)]
pub enum ProvingError {
    /// IR contains an operation not supported in circuit synthesis.
    UnsupportedIR { op_name: String },
    /// Circuit synthesis failed.
    SynthesisError { detail: String },
    /// Proof generation failed.
    ProvingFailed { detail: String },
    /// Proof verification failed.
    VerificationFailed { detail: String },
    /// A required witness value is missing or has the wrong type.
    WitnessMissing { key: String },
    /// The circuit requires more rows than available for the given k.
    CircuitTooLarge {
        needed_rows: usize,
        available_rows: usize,
    },
    /// Generic proving-pipeline error.
    Other { detail: String },
}

impl ProvingError {
    pub fn unsupported(op: &str) -> Self {
        ProvingError::UnsupportedIR {
            op_name: op.to_string(),
        }
    }

    pub fn synthesis(detail: impl Into<String>) -> Self {
        ProvingError::SynthesisError {
            detail: detail.into(),
        }
    }

    pub fn witness_missing(key: impl Into<String>) -> Self {
        ProvingError::WitnessMissing {
            key: key.into(),
        }
    }

    pub fn other(detail: impl Into<String>) -> Self {
        ProvingError::Other {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for ProvingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvingError::UnsupportedIR { op_name } => {
                write!(f, "Unsupported IR operation in circuit: {}", op_name)
            }
            ProvingError::SynthesisError { detail } => {
                write!(f, "Circuit synthesis error: {}", detail)
            }
            ProvingError::ProvingFailed { detail } => {
                write!(f, "Proof generation failed: {}", detail)
            }
            ProvingError::VerificationFailed { detail } => {
                write!(f, "Proof verification failed: {}", detail)
            }
            ProvingError::WitnessMissing { key } => {
                write!(f, "Missing witness value for key: {}", key)
            }
            ProvingError::CircuitTooLarge {
                needed_rows,
                available_rows,
            } => {
                write!(
                    f,
                    "Circuit too large: needs {} rows but only {} available (increase k)",
                    needed_rows, available_rows
                )
            }
            ProvingError::Other { detail } => write!(f, "{}", detail),
        }
    }
}

impl std::error::Error for ProvingError {}

impl From<ProvingError> for PyErr {
    fn from(err: ProvingError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

impl From<String> for ProvingError {
    fn from(s: String) -> Self {
        ProvingError::Other { detail: s }
    }
}

impl From<&str> for ProvingError {
    fn from(s: &str) -> Self {
        ProvingError::Other {
            detail: s.to_string(),
        }
    }
}
