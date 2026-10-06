//! First and last set position of a vector of flags, by running products.
//!
//! `min` / `max` of `np.flatnonzero(m)` (and the per-axis `min(0)` /
//! `max(0)` of `np.argwhere`) are the first and last position whose flag is
//! set. A select chain computes either one at 8 halo2-base cells per flag
//! and leaves nothing a later gather can reuse. The running-product scan
//! below costs about the same, and its by-products are the one-hot of the
//! result and a thermometer (the `IndexEncoding` it records), which a window
//! by that result reads instead of re-deriving them by equality tests and
//! comparisons. No compaction is built and no prover hint is used.
//!
//! `argmax` of a boolean array is the first set position too, but total
//! (0 when none is set); `argmax_of_flags` has that semantics and no
//! precondition beyond 0/1 flags.
//!
//! **Preconditions** (`first_set`, `last_set`). Every flag `has[i]` is a 0/1
//! value (a comparison's result bit). At least one flag is set wherever the
//! result is used: NumPy raises on `min`/`max` of an empty array, and the
//! callers assert non-emptiness first (`require_non_empty`, under the
//! execution predicate).
//!
//! **Postconditions.** The result `v` is the first (last) set position; the
//! recorded encoding's bits are `[v == k]` and `[v < k]`. The fact
//! `0 <= v < n` is recorded too (it holds on every witness, empty included).
//!
//! **Soundness.** Every value below is a polynomial of the flags; no value is
//! free. For `first_set`: `lead[i] = prod_{j <= i} (1 - has[j])` is 1 exactly
//! when no flag in `0..=i` is set (induction on `i`: `lead[i] = lead[i-1] -
//! lead[i-1] * has[i]`), `first[i] = lead[i-1] * has[i]` is 1 exactly at the
//! first set position, and `v = sum_{i < n-1} lead[i]` counts the unset
//! flags before it. `last_set` is the mirror image with `trail[i] = [no
//! flag in i..n is set]`. On an empty vector the one-hot is all zero and
//! `v` is `n - 1` (first) or `0` (last), which only the asserted
//! precondition rules out.
//!
//! **Cost** per flag: halo2-base about 8 cells (one product and one
//! subtraction gate) plus 3 for the result's sum; Noir about 2 gates.

use crate::builder::IRBuilder;
use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};
use crate::types::Value;

use super::encoding::{record_encoding, Bit, IndexEncoding};

/// The first set position of `has` (see the module docs), with its encoding
/// recorded. `None` for an empty vector.
pub fn first_set(b: &mut IRBuilder, has: &[Value]) -> Option<Value> {
    let n = has.len();
    if n == 0 {
        return None;
    }
    let (first, lead) = leading_scan(b, has);
    let value = sum(b, &lead[..n - 1]);
    // [v < k] = 1 - [v >= k] = 1 - lead[k - 1].
    let mut below = vec![Bit::plain(b.ir_constant_int(0))];
    below.extend((1..n).map(|k| Bit::negated(lead[k - 1].clone())));
    finish(b, value, first, below, None)
}

/// NumPy's `argmax` of the 0/1 values `has`: the first set position, or 0
/// when none is set. Total (no precondition): the recorded one-hot is exact
/// on every witness (`[v == 0]` is `has[0]` or "none set"); no thermometer
/// is recorded. Cost as `first_set`.
pub fn argmax_of_flags(b: &mut IRBuilder, has: &[Value]) -> Option<Value> {
    let n = has.len();
    if n == 0 {
        return None;
    }
    let (first, lead) = leading_scan(b, has);
    // v = #{unset flags before the first set one} - (n - 1) [none set].
    let leading_unset = sum(b, &lead[..n - 1]);
    let value = if n == 1 {
        b.ir_constant_int(0)
    } else {
        let n_minus_1 = b.ir_constant_int(n as i64 - 1);
        let none_set = b.ir_mul_i(&lead[n - 1], &n_minus_1);
        b.ir_sub_i(&leading_unset, &none_set)
    };
    let mut onehot = first;
    onehot[0] = b.ir_add_i(&onehot[0], &lead[n - 1]);
    record_range_fact(b, &value, 0, n as i64 - 1);
    record_encoding(b, &value, IndexEncoding { lo: 0, onehot, below: None, last_of: None });
    Some(value)
}

/// `(first, lead)` of `has`: `lead[i] = [no flag in 0..=i is set]`,
/// `first[i] = [i is the first set position]`.
fn leading_scan(b: &mut IRBuilder, has: &[Value]) -> (Vec<Value>, Vec<Value>) {
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

/// The last set position of `has` (see the module docs), with its encoding
/// recorded. `None` for an empty vector.
pub fn last_set(b: &mut IRBuilder, has: &[Value]) -> Option<Value> {
    let n = has.len();
    if n == 0 {
        return None;
    }
    // trail[i] = [no flag in i..n is set], built from the end.
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
    finish(b, value, last, below, Some(has.to_vec()))
}

/// Record the result's encoding and its range fact.
fn finish(
    b: &mut IRBuilder,
    value: Value,
    onehot: Vec<Value>,
    below: Vec<Bit>,
    last_of: Option<Vec<Value>>,
) -> Option<Value> {
    let n = onehot.len() as i64;
    record_range_fact(b, &value, 0, n - 1);
    record_encoding(b, &value, IndexEncoding { lo: 0, onehot, below: Some(below), last_of });
    Some(value)
}

/// `values[0] + ... + values[len-1]` (the constant 0 for none).
fn sum(b: &mut IRBuilder, values: &[Value]) -> Value {
    let mut terms = values.iter();
    let Some(first) = terms.next() else {
        return b.ir_constant_int(0);
    };
    terms.fold(first.clone(), |acc, v| b.ir_add_i(&acc, v))
}

/// The fact `lo <= v <= hi`, which holds on every witness by construction.
fn record_range_fact(b: &mut IRBuilder, v: &Value, lo: i64, hi: i64) {
    let Some(vid) = v.value_id() else { return };
    let var = || Box::new(ContractTerm::Var(ContractVar::Value(vid)));
    let fact = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp { op: CmpOp::Ge, lhs: var(), rhs: Box::new(ContractTerm::LitInt(lo)) },
            ContractTerm::Cmp { op: CmpOp::Le, lhs: var(), rhs: Box::new(ContractTerm::LitInt(hi)) },
        ],
    };
    b.facts.insert_for(vid, fact);
}
