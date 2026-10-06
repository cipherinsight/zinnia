//! The scatter family: operators that move cells to positions that depend
//! on runtime values (insertion, deletion, scatters, compaction).
//!
//! Each construction documents its preconditions, postconditions, cost per
//! backend and its soundness argument next to the code.

pub mod arith;
pub mod insert;
pub mod position;

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod test_support;
