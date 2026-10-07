//! Read-only tables (`IR::RomInit`, `IR::RomRead`): a dynamic lookup into
//! the table's rows (`circuit/tables.rs`).
//!
//! A table's rows are `(tag, address, value)` with the tag and the address
//! fixed and the value copy-constrained to the table's cell; a read is a key
//! row `(tag, index, value)` with the index and the value copy-constrained to
//! the program's cells, and the lookup argument proves every key row is a
//! row of its tag's table. So a read has a witness exactly when the index is
//! one of the table's addresses, and its value is then that cell. The value
//! is a prover hint the lookup pins (tamper test
//! `table_reads_are_pinned_and_in_bounds`).

use crate::prove::error::ProvingError;
use crate::prove::kernel;
use crate::types::StmtId;

use super::ranges::Interval;
use super::{Fp, Halo2BaseCell, Halo2BaseSynthesizer};

/// One table: its tag in the lookup and its cells.
pub(super) struct Rom {
    tag: u64,
    cells: Vec<Halo2BaseCell>,
}

impl Halo2BaseSynthesizer<'_> {
    /// A fresh table tag.
    fn next_table_tag(&mut self) -> u64 {
        let tag = self.table_count;
        self.table_count += 1;
        tag
    }

    /// `RomInit`: register the table's rows under a fresh tag.
    pub(super) fn rom_init_cells(&mut self, table: StmtId, cells: &[Halo2BaseCell]) -> Result<(), ProvingError> {
        let tag = self.next_table_tag();
        for (address, cell) in cells.iter().enumerate() {
            let value = self.assigned(cell);
            self.tables.rows.push((tag, address as u64, value));
        }
        self.roms.insert(table, Rom { tag, cells: cells.to_vec() });
        Ok(())
    }

    /// `RomRead`: a constant address reads its cell (an address outside the
    /// table is a compiler error); a runtime one is a lookup.
    pub(super) fn rom_read_cell(&mut self, table: StmtId, address: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let rom = self.roms.get(&table).ok_or_else(|| ProvingError::synthesis(format!("statement {table} is not a table")))?;
        let (tag, len) = (rom.tag, rom.cells.len());
        let index = kernel::field_position(address.value, len as u64).map(|i| i as usize);
        if self.is_const_cell(address) {
            let index = index.ok_or_else(|| ProvingError::synthesis("table read out of bounds"))?;
            return Ok(self.roms[&table].cells[index]);
        }
        if index.is_none() {
            self.note_unsat(|| format!("Witness does not satisfy circuit constraints: table read at an address outside [0, {len})"));
        }
        let value = index.map_or(Fp::zero(), |i| self.roms[&table].cells[i].value);
        // A prover-supplied value, pinned by the lookup to the cell at `address`.
        let read = self.hint(value);
        let (index_cell, value_cell) = (self.assigned(address), self.assigned(&read));
        self.tables.reads.push((tag, index_cell, value_cell));
        if let Some(range) = Interval::new(0, len as i128 - 1) {
            self.prove_range(address, range);
        }
        Ok(read)
    }
}
