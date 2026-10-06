//! Cost formulas of the memory family's constructions, per target.
//!
//! The scalar unit costs are the gather family's (`gather::cost::units`:
//! one place for the price of an equality test, a select, a multiply-add),
//! plus the index bits below; all were calibrated by compiling reads with
//! each element construction forced (`ZINNIA_FORCE_CONSTRUCTION`) over a
//! grid of line lengths and read counts (`eval-redo/layer/layer-memory.md`,
//! "Calibration").

use crate::lower::gather::cost::units;
use crate::lower::{Context, OpKind, Target};
use crate::types::StmtId;

/// The bound `i < n` with `i`'s `w = ceil(log2 n)` bits
/// (`IR::AssertU128LtBit`): `base + per_bit * w` on each target.
fn bits_cost(n: u64, target: Target) -> u64 {
    let w = u64::from(u64::BITS - (n - 1).leading_zeros());
    let (base, per_bit) = match target {
        Target::Halo2Base => (10, 5),
        Target::Noir => (4, 2),
    };
    base + per_bit * w
}

/// One read of `n >= 2` cells by the multiplexer: `n - 1` selects of two
/// variable branches (`n = 2`: one `select_bit`, the select and the bit
/// assertion).
pub fn index_bits_read(n: u64, target: Target) -> u64 {
    let u = units(target);
    if n == 2 {
        return u.select_var + 1;
    }
    (n - 1) * u.select_var
}

/// The bound and the bits of one index below `n >= 3`.
pub fn index_bits(n: u64, target: Target) -> u64 {
    bits_cost(n, target)
}


/// A table group's fixed cost on halo2-base in cell equivalents: its value
/// column, a key set's two advice columns and the lookup argument's
/// polynomials, about 5 committed polynomials of 2^k rows, at the `k = 9` of
/// the programs this construction pays on (`estimate_params` prices a
/// committed polynomial as 2^k cells).
const TABLE_GROUP_CELLS: u64 = 5 * 512;

/// Table rows a group holds at `k = 9` (`circuit::table_rows(9)` less the
/// free row).
const TABLE_GROUP_ROWS: u64 = 512 - 20 - 1;

/// Reads of a line of `n` cells as table reads, at the lanes' indices with
/// `index_reads` reads each over the program. `offered` groups with
/// `offered_rows` cells in all are offered a table read; the halo2-base
/// table groups those rows need (their fixed cost) are shared among them.
pub fn table(n: u64, index_reads: &[u64], offered: u64, offered_rows: u64, target: Target) -> u64 {
    let reads = index_reads.len() as u64;
    match target {
        // A value cell per slot; per read its value hint and the key row's
        // index and value cells.
        Target::Halo2Base => {
            let fixed = TABLE_GROUP_CELLS * offered_rows.div_ceil(TABLE_GROUP_ROWS).max(1);
            n + 3 * reads + fixed.div_ceil(offered.max(1))
        }
        // The native array (3 gates per element), a ROM read and the index
        // check, which reads at the same index share.
        Target::Noir => {
            let u = units(target);
            n * u.native_element + reads * u.native_read + index_reads.iter().map(|&r| u.native_index.div_ceil(r.max(1))).sum::<u64>()
        }
    }
}

/// For each index, the number of element reads at it over every element
/// group of the program.
pub fn index_reads(cx: &Context, indices: &[StmtId]) -> Vec<u64> {
    indices
        .iter()
        .map(|index| {
            cx.groups
                .iter()
                .filter(|g| g.kind == OpKind::Element)
                .flat_map(|g| &g.lanes)
                .filter(|lane| lane.own.first() == Some(index))
                .count() as u64
        })
        .collect()
}
