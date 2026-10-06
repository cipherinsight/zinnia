//! Memory family of the array-operator lowering layer: reads of an array at
//! a runtime index.
//!
//! The frontend's runtime reads of an array whose cells are known values (a
//! subscript `x[i]`, `a[:, j]` row by row, `np.take_along_axis` over a small
//! axis) emit one placeholder per read through [`read_lane`]: `OpKind::Read`
//! over the cells the read may select (the line along the indexed axis) with
//! the index as the lane's own operand. Reads of the same line form one
//! group; reads of different lines at the same index share whatever a
//! construction builds per index (equality tests, index bits), which the
//! cost formulas account for (`cost::index_share`) and the IR cleanup after
//! lowering deduplicates.
//!
//! Constructions ([`constructions`], each with its preconditions,
//! postconditions, cost and soundness argument):
//!
//! * `ReadBySelectChain`: `[i == k]` per position, a select chain per read;
//! * `ReadByOneHot`: the same tests, exactly one asserted (proves the bound),
//!   an inner product per read;
//! * `ReadByIndexBits`: `i < n` with `i`'s bits (proves the bound), a binary
//!   multiplexer per read;
//! * `ReadByMemory`: the line in a memory segment and the backend's dynamic
//!   read (a native ROM read on Noir; a one-hot or multiplexer on
//!   halo2-base, chosen by the backend per index).
//!
//! The selection picks per group by cost on the compile target
//! (`cost.rs`, calibrated against measured cells and gates; see
//! `eval-redo/layer/layer-memory.md`).
//!
//! Measurement only: the environment variable `ZINNIA_MEMORY_READ` set to a
//! construction's short name (`chain`, `onehot`, `bits`, `memory`) makes it
//! the only construction this family offers, so cost curves and the scaling
//! demonstration can compile the same program with each alternative. Nothing
//! in the benchmark harness sets it.

pub mod constructions;
pub mod cost;
pub mod semantics;

#[cfg(test)]
mod tests;

use std::sync::Arc;

pub use constructions::constructions;
pub use semantics::ReadSemantics;

use crate::builder::IRBuilder;
use crate::lower::{ArrayOp, OpKind};
use crate::types::Value;

/// `cells[index]` as a read placeholder, or `None` when the read needs none
/// (fewer than two cells) or is not a scalar read (complex cells).
///
/// The caller has emitted the program's bounds check `0 <= index <
/// cells.len()` for `index` (the value it passes); constructions whose
/// constraints prove the bound make that check redundant, and the layer then
/// drops it.
pub fn read_lane(b: &mut IRBuilder, cells: &[Value], index: &Value) -> Option<Value> {
    if cells.len() < 2 || cells.iter().any(|c| matches!(c, Value::Complex { .. })) || index.stmt_id().is_none() {
        return None;
    }
    let shared: Arc<[Value]> = cells.iter().map(|c| b.ensure_ptr(c)).collect();
    let op = ArrayOp { kind: OpKind::Read, shared: shared.len() as u32, output: 0, position: None };
    Some(b.ir_array_op(op, &shared, std::slice::from_ref(index)))
}

/// The construction `ZINNIA_MEMORY_READ` forces, if set (measurement only;
/// see the module documentation).
fn forced() -> Option<String> {
    std::env::var("ZINNIA_MEMORY_READ").ok().filter(|s| !s.is_empty())
}
