//! Memory family of the array-operator lowering layer: how a read of an
//! array at a runtime index is built, chosen per read group and target.
//!
//! The reads are the gather family's `OpKind::Element` placeholders
//! (`lower::gather::checked_element_lane`): `x[i]` over the cells `x` of one
//! line (the candidates along the indexed axis), the index `i` checked into
//! `[0, n)` by the frontend where the read runs. The frontend emits one for
//! every runtime read of a known-valued array with one runtime index (an
//! element read `x[i]`, `a[r, j]`, a payload gather `a[:, j]`, a small-axis
//! `np.take_along_axis`). Reads of the same line form a group; reads at the
//! same index share whatever a construction builds per index.
//!
//! The gather family offers the select chain, the inner product with the
//! index's equality bits, the inner product with a recorded one-hot, and the
//! backend memory read (`gather/constructions.rs`). This family adds:
//!
//! * [`ElementByIndexBits`]: a binary multiplexer over the index bits, with
//!   the bound `i < n` proved by the bit decomposition (the program's check
//!   is then dropped);
//!
//! each with its preconditions, postconditions, cost formula (`cost.rs`,
//! calibrated per target) and soundness argument. A halo2-base
//! dynamic-lookup ROM and a shuffle-checked RAM were measured outside the
//! compiler, in `benchmarking/memory_bench`.

pub mod constructions;
pub mod cost;
pub mod versions;
pub mod wrap;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod versions_tests;
#[cfg(test)]
pub(crate) mod test_support;

pub use constructions::{constructions, ElementByIndexBits, ElementByTable};
