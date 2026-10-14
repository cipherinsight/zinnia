//! Static-ndarray (composite) op implementations.
//!
//! Originally a single 4090 LOC file; split into focused submodules for
//! navigability 2026-05-15. The submodules re-export their public
//! surface here so external callers keep using
//! `crate::ops::static_ndarray_ops::np_*` exactly as before.

mod constructors;
mod determinant;
mod diagonals;
mod differences;
mod elementwise;
pub mod einsum;
mod indexing;
mod linalg;
mod matmul;
mod norm;
mod operand;
mod polynomials;
mod products;
mod reductions;
mod reshaping;
mod statistics;

// Re-export every public surface so the path
// `crate::ops::static_ndarray_ops::np_foo` keeps resolving for all
// external callers without changes to their code.
pub use constructors::*;
pub use determinant::np_linalg_det;
pub use diagonals::{np_diagonal, np_tri};
pub use differences::{np_ediff1d, np_trapezoid, TrapezoidSpacing};
pub use elementwise::*;
pub use indexing::*;
pub use linalg::*;
pub use matmul::*;
pub use norm::{np_linalg_norm, NormCall, NormOrder};
pub use polynomials::{np_polyval, np_vander};
pub use products::{np_cross, np_kron, np_vdot};
pub use reductions::*;
pub use reshaping::*;
pub use statistics::*;

#[cfg(test)]
mod tests;
