//! A position of a vector of flags: the first or last set one, or NumPy's
//! `argmax` of booleans.
//!
//! `min` / `max` of `np.flatnonzero(m)` (and the per-axis `min(0)` /
//! `max(0)` of `np.argwhere`) are the first and last position whose flag is
//! set; `argmax` of a boolean array is the first set position, or 0 when
//! none is. Two generators:
//!
//! * [`scan`]: running products. Besides the position it yields the
//!   position's one-hot and a thermometer (an [`IndexEncoding`]), which a
//!   window or an element read at that position consumes instead of
//!   building indicators by equality tests and comparisons.
//! * [`select_chain`]: `select(has[i], i, acc)` from one end to the other,
//!   the position only.
//!
//! **Preconditions.** Every flag `has[i]` is a 0/1 value (a comparison's
//! result bit). For `First` and `Last`, at least one flag is set wherever
//! the result is used: NumPy raises on `min`/`max` of an empty array, and
//! the callers assert non-emptiness first (`require_non_empty`, under the
//! execution predicate). `Argmax` has no further precondition.
//!
//! **Postconditions.** The result `v` is the stated position, in `[0, n)` on
//! every witness; the encoding's bits are `[v == k]` and (`First`, `Last`)
//! `[v < k]`.
//!
//! **Soundness.** No prover hint: every value is a polynomial of the flags.
//! `lead[i] = prod_{j <= i} (1 - has[j])` is 1 exactly when no flag in
//! `0..=i` is set (induction: `lead[i] = lead[i-1] - lead[i-1] * has[i]`),
//! `first[i] = lead[i-1] * has[i]` is 1 exactly at the first set position,
//! and `v = sum_{i < n-1} lead[i]` counts the unset flags before it
//! (`Argmax` subtracts `(n - 1) lead[n-1]`, so `v = 0` when none is set, and
//! adds `lead[n-1]` to `[v == 0]`). `Last` is the mirror image with
//! `trail[i] = [no flag in i..n is set]` and `v = (n-1) - sum_{i >= 1}
//! trail[i]`. On an empty vector the `First`/`Last` one-hot is all zero and
//! `v` is `n - 1` or `0`, which only the asserted precondition rules out.
//!
//! **Cost** (`cost.rs`) per flag: halo2-base about 8 cells for the running
//! products (one product, one subtraction) and 3 for the sum, a select chain
//! about 4 per link; Noir about 2 and 2 gates.

use crate::builder::IRBuilder;
use crate::lower::OpKind;
use crate::types::Value;

use super::encoding::{Bit, IndexEncoding};

/// Which position of the flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanKind {
    /// The first set position (`min` of `flatnonzero`).
    First,
    /// The last set position (`max`).
    Last,
    /// NumPy's `argmax` of the flags: the first set position, 0 when none.
    Argmax,
}

impl ScanKind {
    pub fn of(kind: OpKind) -> Option<ScanKind> {
        match kind {
            OpKind::FirstSet => Some(ScanKind::First),
            OpKind::LastSet => Some(ScanKind::Last),
            OpKind::ArgmaxOfFlags => Some(ScanKind::Argmax),
            _ => None,
        }
    }

    pub fn op_kind(self) -> OpKind {
        match self {
            ScanKind::First => OpKind::FirstSet,
            ScanKind::Last => OpKind::LastSet,
            ScanKind::Argmax => OpKind::ArgmaxOfFlags,
        }
    }

    /// Whether `scan` yields a thermometer of the result.
    pub fn has_thermometer(self) -> bool {
        self != ScanKind::Argmax
    }
}

/// The position by running products, and its encoding. `has` is not empty.
pub fn scan(b: &mut IRBuilder, kind: ScanKind, has: &[Value]) -> (Value, IndexEncoding) {
    let n = has.len();
    assert!(n > 0, "a scan over no flags");
    match kind {
        ScanKind::First | ScanKind::Argmax => {
            let (mut first, lead) = leading(b, has);
            let mut value = sum(b, &lead[..n - 1]);
            let below = if kind == ScanKind::Argmax {
                // None set: v = 0 (not n - 1), and [v == 0] holds.
                if n > 1 {
                    let n_minus_1 = b.ir_constant_int(n as i64 - 1);
                    let none_set = b.ir_mul_i(&lead[n - 1], &n_minus_1);
                    value = b.ir_sub_i(&value, &none_set);
                }
                first[0] = b.ir_add_i(&first[0], &lead[n - 1]);
                None
            } else {
                // [v < k] = 1 - [v >= k] = 1 - lead[k - 1].
                let mut below = vec![Bit::plain(b.ir_constant_int(0))];
                below.extend((1..n).map(|k| Bit::negated(lead[k - 1].clone())));
                Some(below)
            };
            (value, IndexEncoding { lo: 0, onehot: first, below })
        }
        ScanKind::Last => {
            let (last, trail) = trailing(b, has);
            // v = (n - 1) - #{k >= 1 : trail[k]}.
            let n_minus_1 = b.ir_constant_int(n as i64 - 1);
            let value = if n == 1 {
                n_minus_1
            } else {
                let unset_after = sum(b, &trail[1..]);
                b.ir_sub_i(&n_minus_1, &unset_after)
            };
            // [v < k] = [no flag in k..n is set] = trail[k].
            let mut below = vec![Bit::plain(b.ir_constant_int(0))];
            below.extend((1..n).map(|k| Bit::plain(trail[k].clone())));
            (value, IndexEncoding { lo: 0, onehot: last, below: Some(below) })
        }
    }
}

/// The position by a select chain: `select(has[i], i, acc)` from the end
/// for `First` and `Argmax` (so the first set flag wins), from the start for
/// `Last`, starting at 0.
pub fn select_chain(b: &mut IRBuilder, kind: ScanKind, has: &[Value]) -> Value {
    let mut acc = b.ir_constant_int(0);
    let order: Vec<usize> = match kind {
        ScanKind::First | ScanKind::Argmax => (0..has.len()).rev().collect(),
        ScanKind::Last => (0..has.len()).collect(),
    };
    for i in order {
        let position = b.ir_constant_int(i as i64);
        acc = b.ir_select_i(&has[i], &position, &acc);
    }
    acc
}

/// `(first, lead)`: `lead[i] = [no flag in 0..=i is set]`, `first[i] = [i
/// is the first set position]`.
fn leading(b: &mut IRBuilder, has: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let mut first = Vec::with_capacity(has.len());
    let mut lead: Vec<Value> = Vec::with_capacity(has.len());
    for (i, flag) in has.iter().enumerate() {
        if i == 0 {
            first.push(flag.clone());
            let one = b.ir_constant_int(1);
            lead.push(b.ir_sub_i(&one, flag));
        } else {
            let hit = b.ir_mul_i(&lead[i - 1], flag);
            lead.push(b.ir_sub_i(&lead[i - 1], &hit));
            first.push(hit);
        }
    }
    (first, lead)
}

/// `(last, trail)`: `trail[i] = [no flag in i..n is set]`, `last[i] = [i is
/// the last set position]`.
fn trailing(b: &mut IRBuilder, has: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let n = has.len();
    let mut trail: Vec<Value> = vec![b.ir_constant_int(0); n];
    let mut last: Vec<Value> = vec![b.ir_constant_int(0); n];
    for i in (0..n).rev() {
        if i == n - 1 {
            last[i] = has[i].clone();
            let one = b.ir_constant_int(1);
            trail[i] = b.ir_sub_i(&one, &has[i]);
        } else {
            let hit = b.ir_mul_i(&trail[i + 1], &has[i]);
            trail[i] = b.ir_sub_i(&trail[i + 1], &hit);
            last[i] = hit;
        }
    }
    (last, trail)
}

/// `values[0] + ... + values[len-1]` (the constant 0 for none).
fn sum(b: &mut IRBuilder, values: &[Value]) -> Value {
    let mut terms = values.iter();
    let Some(first) = terms.next() else {
        return b.ir_constant_int(0);
    };
    terms.fold(first.clone(), |acc, v| b.ir_add_i(&acc, v))
}
