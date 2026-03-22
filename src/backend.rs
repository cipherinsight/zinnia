//! Backend code generators — stub implementations.
//!
//! Defines the `ProgramBuilder` trait matching Python's `AbstractProgramBuilder`.
//! All 4 backends (Halo2, Circom, Noir, CirC-Zokrates) are stubbed out.
//! They will be replaced with a more robust backend system in a future phase.

use serde::{Deserialize, Serialize};

use crate::ir::IRStatement;

// ---------------------------------------------------------------------------
// Backend enum
// ---------------------------------------------------------------------------

/// The supported ZK backend targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Backend {
    Halo2,
    Circom,
    Noir,
    CirCZokrates,
}

impl Backend {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "halo2" | "HALO2" => Some(Self::Halo2),
            "circom" | "CIRCOM" => Some(Self::Circom),
            "noir" | "NOIR" => Some(Self::Noir),
            "circ_zok" | "CIRC_ZOK" | "circ_zokrates" => Some(Self::CirCZokrates),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Backend::Halo2 => "halo2",
            Backend::Circom => "circom",
            Backend::Noir => "noir",
            Backend::CirCZokrates => "circ_zok",
        }
    }
}

// ---------------------------------------------------------------------------
// ProgramBuilder trait
// ---------------------------------------------------------------------------

/// Trait matching Python `AbstractProgramBuilder`.
/// Takes a circuit name and a list of IR statements, produces compiled output.
pub trait ProgramBuilder {
    /// Build the compiled output from the IR statements.
    /// Returns the generated source code string.
    fn build(&self) -> String;

    /// The backend this builder targets.
    fn backend(&self) -> Backend;
}

// ---------------------------------------------------------------------------
// Stub implementations
// ---------------------------------------------------------------------------

pub struct Halo2ProgramBuilder {
    pub name: String,
    pub stmts: Vec<IRStatement>,
}

impl ProgramBuilder for Halo2ProgramBuilder {
    fn build(&self) -> String {
        unimplemented!(
            "Halo2 backend not yet implemented in Rust (circuit: {}, {} stmts). \
             Will be replaced with a robust backend.",
            self.name,
            self.stmts.len()
        )
    }

    fn backend(&self) -> Backend {
        Backend::Halo2
    }
}

pub struct CircomProgramBuilder {
    pub name: String,
    pub stmts: Vec<IRStatement>,
}

impl ProgramBuilder for CircomProgramBuilder {
    fn build(&self) -> String {
        unimplemented!(
            "Circom backend not yet implemented in Rust (circuit: {}, {} stmts)",
            self.name,
            self.stmts.len()
        )
    }

    fn backend(&self) -> Backend {
        Backend::Circom
    }
}

pub struct NoirProgramBuilder {
    pub name: String,
    pub stmts: Vec<IRStatement>,
}

impl ProgramBuilder for NoirProgramBuilder {
    fn build(&self) -> String {
        unimplemented!(
            "Noir backend not yet implemented in Rust (circuit: {}, {} stmts)",
            self.name,
            self.stmts.len()
        )
    }

    fn backend(&self) -> Backend {
        Backend::Noir
    }
}

pub struct CirCZokratesProgramBuilder {
    pub name: String,
    pub stmts: Vec<IRStatement>,
}

impl ProgramBuilder for CirCZokratesProgramBuilder {
    fn build(&self) -> String {
        unimplemented!(
            "CirC-Zokrates backend not yet implemented in Rust (circuit: {}, {} stmts)",
            self.name,
            self.stmts.len()
        )
    }

    fn backend(&self) -> Backend {
        Backend::CirCZokrates
    }
}

// ---------------------------------------------------------------------------
// Dispatch function
// ---------------------------------------------------------------------------

/// Create a backend builder for the given backend type.
pub fn create_builder(
    backend: Backend,
    name: String,
    stmts: Vec<IRStatement>,
) -> Box<dyn ProgramBuilder> {
    match backend {
        Backend::Halo2 => Box::new(Halo2ProgramBuilder { name, stmts }),
        Backend::Circom => Box::new(CircomProgramBuilder { name, stmts }),
        Backend::Noir => Box::new(NoirProgramBuilder { name, stmts }),
        Backend::CirCZokrates => Box::new(CirCZokratesProgramBuilder { name, stmts }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_from_str() {
        assert_eq!(Backend::parse("halo2"), Some(Backend::Halo2));
        assert_eq!(Backend::parse("CIRCOM"), Some(Backend::Circom));
        assert_eq!(Backend::parse("noir"), Some(Backend::Noir));
        assert_eq!(Backend::parse("circ_zok"), Some(Backend::CirCZokrates));
        assert_eq!(Backend::parse("unknown"), None);
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(Backend::Halo2.name(), "halo2");
        assert_eq!(Backend::Circom.name(), "circom");
        assert_eq!(Backend::Noir.name(), "noir");
        assert_eq!(Backend::CirCZokrates.name(), "circ_zok");
    }

    #[test]
    fn test_create_builder() {
        let builder = create_builder(Backend::Halo2, "test".to_string(), vec![]);
        assert_eq!(builder.backend(), Backend::Halo2);
    }
}
