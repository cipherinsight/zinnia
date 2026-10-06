//! Cost formulas of the memory family's read constructions, per target.
//!
//! Unit costs are measured averages of one scalar operation in the shape
//! the emitters produce it (halo2-base advice + lookup cells at lookup width
//! 8, Noir `bb gates`), calibrated by compiling the same reads with each
//! construction forced (`ZINNIA_MEMORY_READ`) at several line lengths and
//! read counts; the measurements are in `eval-redo/layer/layer-memory.md`.

use crate::lower::{Context, Group, OpKind, Target};
use crate::types::StmtId;

/// A read construction (see `constructions.rs`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadPlan {
    SelectChain,
    OneHot,
    IndexBits,
    Memory,
}

impl ReadPlan {
    /// The name `ZINNIA_MEMORY_READ` uses for it.
    pub fn short_name(self) -> &'static str {
        match self {
            ReadPlan::SelectChain => "chain",
            ReadPlan::OneHot => "onehot",
            ReadPlan::IndexBits => "bits",
            ReadPlan::Memory => "memory",
        }
    }

    /// Whether its constraints prove `0 <= i < n` for every index it reads
    /// on every backend. The backend memory read does on halo2-base and Noir
    /// but not on the mock (an address past the end reads the segment's
    /// initial value there), so it keeps the program's check.
    pub fn proves_bounds(self) -> bool {
        matches!(self, ReadPlan::OneHot | ReadPlan::IndexBits)
    }
}

/// Cost of the scalar operations the read constructions emit.
#[derive(Clone, Copy)]
struct Units {
    /// `[i == k]` for a constant `k`.
    eq_const: u64,
    /// `select(c, a, b)` of two cells.
    select: u64,
    /// `select_bit(i, a, b)` (asserts the bit).
    select_bit: u64,
    /// `acc + bit * x` in an inner product.
    mul_add: u64,
    /// `a + b`.
    add: u64,
    /// `assert(s == 1)`.
    assert_one: u64,
    /// The bound and bits of an index below `n`: `base + per_bit * w`.
    bits_base: u64,
    bits_per_bit: u64,
    /// Noir native memory: one cell of the array (once per line), one read,
    /// and the index check (once per distinct index).
    native_element: u64,
    native_read: u64,
    native_index: u64,
}

fn units(target: Target) -> Units {
    match target {
        Target::Halo2Base => Units {
            eq_const: 7,
            select: 7,
            select_bit: 5,
            mul_add: 3,
            add: 3,
            assert_one: 1,
            bits_base: 10,
            bits_per_bit: 5,
            native_element: 0,
            native_read: 0,
            native_index: 0,
        },
        Target::Noir => Units {
            eq_const: 3,
            select: 1,
            select_bit: 2,
            mul_add: 1,
            add: 1,
            assert_one: 1,
            bits_base: 4,
            bits_per_bit: 2,
            native_element: 3,
            native_read: 3,
            native_index: 8,
        },
    }
}

/// `ceil(log2 n)` for `n >= 2`.
fn width(n: u64) -> u64 {
    u64::from(u64::BITS - (n - 1).leading_zeros())
}

/// Cost of the per-index part of `plan` (built once per distinct index,
/// shared by every read at it) and of one read, for a line of `n` cells.
fn parts(plan: ReadPlan, n: u64, target: Target) -> (u64, u64) {
    let u = units(target);
    match plan {
        ReadPlan::SelectChain => ((n - 1) * u.eq_const, (n - 1) * u.select),
        ReadPlan::OneHot => (n * u.eq_const + (n - 1) * u.add + u.eq_const + u.assert_one, n * u.mul_add),
        ReadPlan::IndexBits if n == 2 => (0, u.select_bit),
        ReadPlan::IndexBits => (u.bits_base + u.bits_per_bit * width(n), (n - 1) * u.select),
        ReadPlan::Memory => match target {
            // The backend's one-hot gather (`synthesizer/memory.rs`).
            Target::Halo2Base => (n * u.eq_const + n * u.add + u.assert_one, n * u.mul_add),
            Target::Noir => (u.native_index, u.native_read),
        },
    }
}

/// For each read of `group`, the number of reads at its index over every
/// read group of the program (they share the per-index part).
pub fn index_share(group: &Group, cx: &Context) -> Vec<u64> {
    let index_of = |lane: &crate::lower::Lane| -> StmtId { lane.own[0] };
    group
        .lanes
        .iter()
        .map(|lane| {
            let index = index_of(lane);
            cx.groups
                .iter()
                .filter(|g| g.kind == OpKind::Read)
                .flat_map(|g| &g.lanes)
                .filter(|other| index_of(other) == index)
                .count()
                .max(1) as u64
        })
        .collect()
}

/// Cost of producing every read of `group` by `plan`: per read its own
/// part and its share of its index's part, plus the per-line part (Noir's
/// native array).
pub fn read_group(plan: ReadPlan, group: &Group, cx: &Context) -> u64 {
    let n = group.shared.len() as u64;
    let (per_index, per_read) = parts(plan, n, cx.target);
    let per_line = match (plan, cx.target) {
        (ReadPlan::Memory, Target::Noir) => n * units(cx.target).native_element,
        _ => 0,
    };
    let reads: u64 = index_share(group, cx).iter().map(|share| per_read + per_index.div_ceil(*share)).sum();
    per_line + reads
}
