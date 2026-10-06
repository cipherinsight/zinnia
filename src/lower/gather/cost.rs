//! Cost formulas of the gather family's constructions, per target.
//!
//! Unit costs are the measured average cost of one scalar operation in the
//! shape the constructions emit it (halo2-base advice + lookup cells at the
//! perf cases' lookup widths; Noir `bb gates`). They only need to order the
//! alternatives correctly; near-ties were checked by emitting both.

use crate::lower::Target;

/// Cost of the scalar operations the constructions emit.
#[derive(Clone, Copy)]
pub(super) struct Units {
    /// `a * b` as its own gate (its result is a cell other gates read).
    pub mul: u64,
    /// `acc + a * b` (or `acc - a * b`) continuing a chain.
    pub mul_add: u64,
    /// `a + b`, `a - b`.
    pub add: u64,
    /// `[x == k]` for a constant `k`.
    pub eq_const: u64,
    /// `select(c, k, acc)` with a constant branch.
    pub select: u64,
    /// `select(c, x, acc)` with two variable branches.
    pub select_var: u64,
    /// `[x < y]` of two small bounded values.
    pub less_than: u64,
    /// A native array read at an index already checked (Noir; halo2-base
    /// has none): the ROM read and its `u32` address.
    pub native_read: u64,
    /// The range check of one distinct index of native reads, shared by
    /// every read at that index (the Noir emitter checks each index once).
    pub native_index: u64,
    /// One element of a native array, built once per array.
    pub native_element: u64,
}

pub(super) fn units(target: Target) -> Units {
    match target {
        Target::Halo2Base => Units {
            mul: 4,
            mul_add: 3,
            add: 3,
            eq_const: 7,
            select: 7,
            select_var: 4,
            less_than: 20,
            native_read: u64::MAX / 4,
            native_index: 0,
            native_element: 0,
        },
        Target::Noir => Units {
            mul: 1,
            mul_add: 1,
            add: 1,
            eq_const: 3,
            select: 1,
            select_var: 2,
            less_than: 8,
            native_read: 3,
            native_index: 8,
            native_element: 3,
        },
    }
}

/// The running-product scan over `n` flags (`scan.rs`): one
/// multiply-subtract gate per flag (`lead[i] = lead[i-1] - lead[i-1]
/// has[i]`) and the result's sum. The one-hot bits `lead[i-1] has[i]` are
/// then not cells; a consumer of the encoding pays for them
/// (`onehot_cells`).
pub(super) fn scan_running_products(n: u64, argmax: bool, target: Target) -> u64 {
    let u = units(target);
    // argmax: `(n - 1) lead[n-1]` off the sum and `lead[n-1]` onto [v == 0].
    let none_set = if argmax { u.mul + u.add } else { 0 };
    n * u.mul + n.saturating_sub(1) * u.add + none_set
}

/// The scan's one-hot bits as cells, paid by a consumer of its encoding.
pub(super) fn onehot_cells(n: u64, target: Target) -> u64 {
    n * units(target).mul
}

/// A select chain over `n` flags.
pub(super) fn scan_select_chain(n: u64, target: Target) -> u64 {
    n * units(target).select
}

/// How one axis of a window learns its selectors.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct AxisCost {
    pub n: u64,
    /// The start's one-hot is recorded (no equality tests).
    pub start_free: bool,
    pub mask: MaskCost,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum MaskCost {
    /// No mask (`stop >= n`, or the padding is 0 anyway).
    None,
    /// A recorded thermometer of `stop`: one product-subtraction per selector.
    Thermometer,
    /// One comparison per output position, one product per selector.
    Compared,
}

/// The window shift over `dims` (`window.rs`): per axis its selectors and
/// one multiply-add per term of every output cell.
pub(super) fn window_shift(dims: &[u64], axes: &[AxisCost], target: Target) -> u64 {
    let u = units(target);
    let cells: u64 = dims.iter().product();
    axes.iter()
        .map(|a| {
            let pairs = a.n * (a.n + 1) / 2;
            let onehot = if a.start_free { onehot_cells(a.n, target) } else { a.n * u.eq_const };
            let mask = match a.mask {
                MaskCost::None => 0,
                MaskCost::Thermometer => pairs * u.mul,
                MaskCost::Compared => a.n * (u.less_than + u.add) + pairs * u.mul,
            };
            // Each output cell along the axis at position u sums n - u terms.
            let terms = cells / a.n * pairs;
            onehot + mask + terms * u.mul_add
        })
        .sum()
}

/// `x[i]` over `n` cells as an inner product with a recorded one-hot (a
/// scan's bits become cells for it, a scatter's hits are cells already).
pub(super) fn element_by_onehot(n: u64, onehot_is_free: bool, target: Target) -> u64 {
    let bits = if onehot_is_free { 0 } else { onehot_cells(n, target) };
    bits + n * units(target).mul_add
}

/// One read of `n` cells at an index read `index_reads` times by element
/// reads in the whole program (they share its equality tests): a select
/// chain over the index's bits `[i == 1..n)` (`element::gather_by_select_chain`).
/// `checked_below_n`: the program checks `AssertU128Lt(i, n)`. halo2-base
/// then turns a chain of at least three links into an inner product, one
/// multiply-add per link (`prove/halo2_base/synthesizer/plan.rs`,
/// `OneHotChain`); otherwise each link is a select gate.
pub(super) fn element_by_select_chain(n: u64, index_reads: u64, checked_below_n: bool, target: Target) -> u64 {
    let u = units(target);
    let links = n.saturating_sub(1);
    let fused = target == Target::Halo2Base && checked_below_n && links >= 3;
    let link = if fused { u.mul_add } else { u.select_var };
    (links * u.eq_const).div_ceil(index_reads.max(1)) + links * link
}

/// One read of `n` cells at an index read `index_reads` times (see
/// `element_by_select_chain`): the inner product with the bits `[i == 0..n)`
/// (`element::gather_by_indicators`).
pub(super) fn element_by_indicators(n: u64, index_reads: u64, target: Target) -> u64 {
    let u = units(target);
    (n * u.eq_const).div_ceil(index_reads.max(1)) + n * u.mul_add
}

/// A permutation scatter of `n` values: the hits (shared with the
/// permutation precondition, which emits the same tests) and one
/// multiply-add per hit.
pub(super) fn permutation_scatter(n: u64, target: Target) -> u64 {
    n * n * units(target).mul_add
}

/// `reads` reads of `n` cells through memory at the distinct indices
/// `index_reads` (for each, the number of element reads at that index in the
/// whole program, which share its check): halo2-base builds the indicators
/// of each distinct index and an inner product per read; Noir reads a native
/// array, checking each index once (measured on 0319: 3 gates per read at a
/// checked index, 3 per array element, about 8 per index check).
pub(super) fn element_by_memory(n: u64, reads: u64, index_reads: &[u64], target: Target) -> u64 {
    let u = units(target);
    let indices = index_reads.len() as u64;
    match target {
        Target::Halo2Base => indices * n * u.eq_const + reads * n * u.mul_add,
        Target::Noir => {
            let checks: u64 = index_reads.iter().map(|&r| u.native_index.div_ceil(r.max(1))).sum();
            n * u.native_element + reads * u.native_read + checks
        }
    }
}
