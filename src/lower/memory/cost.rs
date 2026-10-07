//! Cost formulas of the memory family's constructions, per target.
//!
//! Unit costs are measured: reads compiled with each element construction
//! forced (`ZINNIA_FORCE_CONSTRUCTION`) at several line lengths
//! (`eval-redo/layer/layer-memory-data/scale.py`, 169 reads at distinct
//! indices over lines of 2..16 cells): a halo2-base select of two variable
//! branches in the multiplexer costs 8 cells, the bound and `w` index bits
//! `4 + 7w` (`7w` when `n` is a power of two), a `select_bit` 12; on Noir 2
//! gates per select, the bits `4 + 2w` (`2w`), a `select_bit` 3. halo2-base
//! costs move with `k` (a range check's lookup cells depend on the lookup
//! width `k - 1`); the formulas fit `k = 7..9`. The Noir native array's costs are the gather
//! family's (`gather::cost::units`).

use crate::lower::gather::cost::units;
use crate::lower::{Context, Group, OpKind, Target};
use crate::types::StmtId;

/// `ceil(log2 n)` for `n >= 2`.
fn width(n: u64) -> u64 {
    u64::from(u64::BITS - (n - 1).leading_zeros())
}

/// One read of `n >= 2` cells by the multiplexer: `n - 1` selects of two
/// variable branches; for `n = 2` one `select_bit` (the select and the bit
/// assertion).
pub fn index_bits_read(n: u64, target: Target) -> u64 {
    let (select, select_bit) = match target {
        Target::Halo2Base => (8, 12),
        Target::Noir => (2, 3),
    };
    if n == 2 {
        return select_bit;
    }
    (n - 1) * select
}

/// The bound `i < n` and `i`'s `w = ceil(log2 n)` bits
/// (`IR::AssertU128LtBit`) of one index below `n >= 3`. For `n = 2^w` the
/// `w` bits are the bound; otherwise both backends add the check
/// `n - 1 - i < 2^w` (`base`).
pub fn index_bits(n: u64, target: Target) -> u64 {
    let (base, per_bit) = match target {
        Target::Halo2Base => (4, 7),
        Target::Noir => (4, 2),
    };
    let bound = if n.is_power_of_two() { 0 } else { base };
    bound + per_bit * width(n)
}

/// A table group's fixed cost on halo2-base in cell equivalents: its value
/// column, a key set's two advice columns and the lookup argument's
/// polynomials, about 5 committed polynomials of 2^k rows, at the `k = 9` of
/// the programs this construction pays on (`estimate_params` prices a
/// committed polynomial as 2^k cells).
pub const TABLE_GROUP_CELLS: u64 = 5 * 512;

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

/// For each index, the number of element reads at it over the element
/// groups that share costs with `group` (`Context::sharing_groups`).
pub fn index_reads(cx: &Context, group: &Group, indices: &[StmtId]) -> Vec<u64> {
    indices
        .iter()
        .map(|index| {
            cx.sharing_groups(group)
                .filter(|g| g.kind == OpKind::Element)
                .flat_map(|g| &g.lanes)
                .filter(|lane| lane.own.first() == Some(index))
                .count() as u64
        })
        .collect()
}

/// NumPy's wrap by a pinned sign (`wrap.rs`): the hint, its bit check, the
/// shift and the select, and `hits` equality tests.
pub fn wrap_by_pinned_sign(hits: u64, target: Target) -> u64 {
    let u = crate::lower::gather::cost::element_units(target);
    let sign = match target {
        Target::Halo2Base => 14,
        Target::Noir => 3,
    };
    sign + hits * u.eq_const
}

/// NumPy's wrap by the pinned position one-hot (`wrap.rs`,
/// `scatter::position`): per position a hinted bit, its bit check, a
/// running sum and a multiply-add; the relation once. Measured against the
/// pinned sign (`layer-memory-data/wrapcal.py`: 16 and 48 reads of a line
/// of n = 2..32 cells, each at its own index, read by the multiplexer or a
/// table under either wrap): about `10n - 8` cells and `1.75n + 1` gates
/// more than the pinned sign's 14 cells and 3 gates.
pub fn wrap_by_position_hits(n: u64, target: Target) -> u64 {
    match target {
        Target::Halo2Base => 10 * n + 6,
        Target::Noir => (7 * n + 16).div_ceil(4),
    }
}
