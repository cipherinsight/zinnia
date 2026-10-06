//! Cost formulas of the memory family's constructions, per target.
//!
//! The scalar unit costs are the gather family's (`gather::cost::units`:
//! one place for the price of an equality test, a select, a multiply-add),
//! plus the index bits below; all were calibrated by compiling reads with
//! each element construction forced (`ZINNIA_FORCE_CONSTRUCTION`) over a
//! grid of line lengths and read counts (`eval-redo/layer/layer-memory.md`,
//! "Calibration").

use crate::lower::gather::cost::units;
use crate::lower::Target;

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
