//! Static-ndarray (composite) op implementations.
//!
//! Originally a single 4090 LOC file; split into focused submodules for
//! navigability 2026-05-15. The submodules re-export their public
//! surface here so external callers keep using
//! `crate::ops::static_ndarray_ops::np_*` exactly as before.

mod constructors;
mod elementwise;
pub mod einsum;
mod indexing;
mod linalg;
mod matmul;
mod reductions;
mod reshaping;
mod statistics;

// Re-export every public surface so the path
// `crate::ops::static_ndarray_ops::np_foo` keeps resolving for all
// external callers without changes to their code.
pub use constructors::*;
pub use elementwise::*;
pub use indexing::*;
pub use linalg::*;
pub use matmul::*;
pub use reductions::*;
pub use reshaping::*;
pub use statistics::*;

#[cfg(test)]
mod tests;
