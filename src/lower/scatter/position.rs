//! The position a runtime index names on an axis, as a pinned one-hot vector.
//!
//! NumPy reads an index `x` on an axis of length `n` from the end when it is
//! negative: `x` names position `x` for `0 <= x < n` and `x + n` for
//! `-n <= x < 0`; any other `x` raises `IndexError`. `np.insert` also accepts
//! `x = n` (insert after the last element). The constructions of this family
//! (scatters, insertion, zeroing writes) all need that position as a one-hot
//! vector `hot[p] = [P == p]` and its thermometer `below[k] = [P < k]`.
//!
//! # Construction: hinted one-hot ([`position_one_hot`])
//!
//! The prover supplies `hot` (`HintKind::PositionOneHot`, one lane per
//! position) and the IR asserts
//!
//! 1. every `hot[p]` is a bit;
//! 2. `sum_p hot[p] = 1` (the last of the running sums `below`);
//! 3. with `P = sum_p p * hot[p]`: `(x - P) * (x - P + w) = 0`, where
//!    `w = n` for an element position and `w = n * (1 - hot[n])` for an
//!    insertion point.
//!
//! **Preconditions**: `n` is a compile-time constant; `x` is an integer (any
//! field element; no window or bound is needed).
//! **Postconditions**: `hot` is the one-hot of NumPy's position of `x`;
//! `below[k] = [P < k]` for `k` in `0..=lanes`; the fact that `x` names a
//! position (`-n <= x < n`, or `<= n` for an insertion point): the circuit
//! has no witness otherwise, as NumPy raises.
//!
//! **Soundness and completeness.** By 1 and 2 exactly one `hot[q]` is 1, so
//! `P = q` is a position (`0 <= q < lanes`) and `below` is its thermometer.
//! The product in 3 is zero in the field exactly when a factor is (a field
//! has no zero divisors), so `x = q` or `x = q - w` as field elements; `x`,
//! `q` and `w` are small integers, so these are integer equalities.
//! * Element position (`w = n`, `q < n`): `x = q` needs `x >= 0`, `x = q - n`
//!   needs `x < 0`, so `q` is `x` or `x + n`, NumPy's position, and there is
//!   no `q` when `x` lies outside `[-n, n)`.
//! * Insertion point: `q = n` gives `w = 0` and `x = n`; `q < n` gives
//!   `w = n` and `x` in `{q, q - n}`, which lies in `[-n, n)`. So `x = n`
//!   admits only `q = n`, `x` in `[0, n)` only `q = x`, `x` in `[-n, 0)` only
//!   `q = x + n`, and any other `x` no `q`.
//!
//! In both cases the honest one-hot satisfies 1-3, and it is the only
//! assignment that does: every hint lane is pinned.
//!
//! **Cost** (per position `p`; `lanes` positions): halo2-base about
//! 11 cells (the hint cell, a 4-cell bit gate, one chained addition for
//! `below`, one chained multiply-add for `P`) plus about 15 cells once for
//! the relation 3; Noir: one gate for the bit check per position (the sums
//! are linear), plus a few gates once.
//!
//! Every check is emitted under the builder's execution predicate
//! (`builder::guard`), so it holds only where the code runs.

use crate::builder::IRBuilder;
use crate::ir_defs::HintKind;
use crate::types::Value;

/// Which positions an index may name on an axis of length `n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Positions {
    /// An element, `-n <= x < n` (`a[x]`, `a[x] = v`): `n` positions.
    Element,
    /// An insertion point, `-n <= x <= n` (`np.insert`): `n + 1` positions.
    InsertionPoint,
}

impl Positions {
    /// The number of positions on an axis of length `n`.
    pub fn count(self, n: usize) -> usize {
        match self {
            Positions::Element => n,
            Positions::InsertionPoint => n + 1,
        }
    }
}

/// NumPy's position `P` of a runtime index, as IR values.
pub struct PositionOneHot {
    /// `hot[p] = [P == p]`, one per position.
    pub hot: Vec<Value>,
    /// `below[k] = [P < k]` for `k` in `0..=hot.len()` (`below[0] = 0`,
    /// and the last is 1).
    pub below: Vec<Value>,
}

impl PositionOneHot {
    /// `[P >= k]`, i.e. `1 - below[k]`, as running sums from the end:
    /// `at_or_above[k] = sum_{p >= k} hot[p]` for `k` in `0..=hot.len()`.
    /// One addition per position, emitted as one chain.
    pub fn at_or_above(&self, b: &mut IRBuilder) -> Vec<Value> {
        let lanes = self.hot.len();
        let mut sums = vec![b.ir_constant_int(0); lanes + 1];
        for k in (0..lanes).rev() {
            sums[k] = b.ir_add_i(&sums[k + 1], &self.hot[k]);
        }
        sums
    }
}

/// The hinted one-hot of NumPy's position of `x` on an axis of length `n`
/// (see the module documentation for the constraints and their argument).
pub fn position_one_hot(b: &mut IRBuilder, x: &Value, n: usize, positions: Positions) -> PositionOneHot {
    let lanes = positions.count(n);
    let axis_len = b.ir_constant_int(n as i64);
    let hot: Vec<Value> = (0..lanes)
        .map(|p| {
            let lane = b.ir_constant_int(p as i64);
            let bit = b.ir_hint(HintKind::PositionOneHot, &[x.clone(), axis_len.clone()], &lane);
            b.ir_assert_bit(&bit);
            bit
        })
        .collect();

    // 2: the running sums end at 1.
    let mut below = Vec::with_capacity(lanes + 1);
    below.push(b.ir_constant_int(0));
    for bit in &hot {
        let next = b.ir_add_i(below.last().expect("below starts with 0"), bit);
        below.push(next);
    }
    let one = b.ir_constant_int(1);
    let single = b.ir_equal_i(below.last().expect("below starts with 0"), &one);
    b.ir_assert(&single);

    // 3: x is P, or P - w.
    let mut position = b.ir_constant_int(0);
    for (p, bit) in hot.iter().enumerate().skip(1) {
        let weight = b.ir_constant_int(p as i64);
        let term = b.ir_mul_i(bit, &weight);
        position = b.ir_add_i(&position, &term);
    }
    let wrap = match positions {
        Positions::Element => axis_len.clone(),
        Positions::InsertionPoint => {
            let at_end = b.ir_mul_i(&hot[n], &axis_len);
            b.ir_sub_i(&axis_len, &at_end)
        }
    };
    let offset = b.ir_sub_i(x, &position);
    let wrapped = b.ir_add_i(&offset, &wrap);
    let product = b.ir_mul_i(&offset, &wrapped);
    let zero = b.ir_constant_int(0);
    let names_position = b.ir_equal_i(&product, &zero);
    b.ir_assert(&names_position);

    PositionOneHot { hot, below }
}
