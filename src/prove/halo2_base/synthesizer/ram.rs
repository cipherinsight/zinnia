//! Read-write memories (`IR::RamInit`, `IR::RamWrite`, `IR::RamRead`): rows
//! of one shared RAM trace (`circuit/ram.rs`).
//!
//! Every memory of the program lives in the one trace, memory `t` at the
//! addresses `[base_t, base_t + n_t)`: its `RamInit` adds one initial row
//! per cell (time 0, the fixed address `base_t + j`, the cell), and each
//! access adds the next row in statement order (the interpreter checks the
//! RAM discipline, so statement order is the memories' access order): a
//! write `(t, write, base_t + a, value)`, a read `(t, read, base_t + a,
//! hint)`. The trace pins every read to the last earlier write at its
//! address (or the initial cell) and proves every trace address in `[0,
//! N)`. That alone would let `a` outside `[0, n_t)` reach another memory's
//! cells, so each access at a runtime address also asserts `a < n_t`
//! (`assert_less`; free when the program's index check already proved it):
//! an address outside its memory has no witness here, as on every backend.
//!
//! So the trace's fixed columns and arguments are paid once per program
//! (`lower::challenge::RAM`), and each memory adds its rows.

use std::collections::HashMap;

use crate::prove::error::ProvingError;
use crate::prove::kernel;
use crate::types::StmtId;

use super::super::circuit::ram::{RamAddress, RamRow, RamTrace};
use super::ranges::Operand;
use super::{Fp, Halo2BaseCell, Halo2BaseSynthesizer};

/// The memories of one synthesis: which trace they share, the next access
/// time, and each memory version's place.
#[derive(Default)]
pub(super) struct Memories {
    /// The index of the shared trace in `Halo2BaseSynthesizer::ram`.
    trace: Option<usize>,
    next_time: u64,
    /// Every memory version (its `RamInit` or `RamWrite` statement): its
    /// memory's base address and contents (what the honest prover reads).
    versions: HashMap<StmtId, Memory>,
}

#[derive(Clone)]
struct Memory {
    base: u64,
    contents: Vec<Fp>,
}

impl Halo2BaseSynthesizer<'_> {
    /// `RamInit`: a memory at the next free addresses of the shared trace,
    /// one initial row per cell.
    pub(super) fn ram_init_cells(&mut self, memory: StmtId, cells: &[Halo2BaseCell]) -> Result<(), ProvingError> {
        let trace = self.memory_trace();
        let base = self.ram.traces[trace].slots;
        for (j, cell) in cells.iter().enumerate() {
            let value = self.assigned(cell);
            self.ram.traces[trace].rows.push(RamRow { time: 0, read: false, address: RamAddress::Fixed(base + j as u64), value });
        }
        self.ram.traces[trace].slots += cells.len() as u64;
        let contents = cells.iter().map(|c| c.value).collect();
        self.memories.versions.insert(memory, Memory { base, contents });
        Ok(())
    }

    /// `RamWrite`: the next row, a write of `value` at `address`.
    pub(super) fn ram_write_cell(&mut self, from: StmtId, to: StmtId, address: &Halo2BaseCell, value: &Halo2BaseCell) -> Result<(), ProvingError> {
        let mut memory = self.memory_version(from)?;
        let index = self.ram_row(&memory, address, value, false)?;
        if let Some(i) = index {
            memory.contents[i] = value.value;
        }
        self.memories.versions.insert(to, memory);
        Ok(())
    }

    /// `RamRead`: the next row, a read at `address` whose value is a hint the
    /// trace pins.
    pub(super) fn ram_read_cell(&mut self, version: StmtId, address: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let memory = self.memory_version(version)?;
        let len = memory.contents.len() as u64;
        let current = kernel::field_position(address.value, len).map_or(Fp::zero(), |i| memory.contents[i as usize]);
        let read = self.hint(current);
        self.ram_row(&memory, address, &read, true)?;
        Ok(read)
    }

    /// The trace every memory shares, created at the first `RamInit`.
    fn memory_trace(&mut self) -> usize {
        if let Some(trace) = self.memories.trace {
            return trace;
        }
        self.ram.traces.push(RamTrace { slots: 0, rows: Vec::new() });
        self.memories.next_time = 1;
        let trace = self.ram.traces.len() - 1;
        self.memories.trace = Some(trace);
        trace
    }

    fn memory_version(&self, version: StmtId) -> Result<Memory, ProvingError> {
        self.memories
            .versions
            .get(&version)
            .cloned()
            .ok_or_else(|| ProvingError::synthesis(format!("statement {version} is not a memory version")))
    }

    /// Append the access row of `memory` at `address` with `value`; the
    /// address's position in the memory (`None` outside it: no witness).
    fn ram_row(&mut self, memory: &Memory, address: &Halo2BaseCell, value: &Halo2BaseCell, read: bool) -> Result<Option<usize>, ProvingError> {
        let len = memory.contents.len();
        let index = kernel::field_position(address.value, len as u64).map(|i| i as usize);
        let address = if self.is_const_cell(address) {
            let index = index.ok_or_else(|| ProvingError::synthesis("memory access out of bounds"))?;
            RamAddress::Fixed(memory.base + index as u64)
        } else {
            if index.is_none() {
                self.note_unsat(|| format!("Witness does not satisfy circuit constraints: memory access at an address outside [0, {len})"));
            }
            // The trace proves only the global address; this memory's own
            // bound is asserted here (no witness for `a` outside `[0, len)`).
            self.assert_less(address, Operand::Constant(len as i128))?;
            RamAddress::Cell { base: memory.base, cell: self.assigned(address) }
        };
        let row = RamRow { time: self.memories.next_time, read, address, value: self.assigned(value) };
        self.memories.next_time += 1;
        let trace = self.memories.trace.expect("a memory exists");
        self.ram.traces[trace].rows.push(row);
        Ok(index)
    }
}
