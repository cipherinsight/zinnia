//! The stable sort by hinted sorted keys and a grand-product permutation
//! check (`Plan::GrandProduct`, construction [`super::GrandProductPermutation`]).
//!
//! Construction.
//! * Preconditions: the sort's elements (int64 values or Q32 codes; the
//!   backend checks each into the signed 64-bit window when it packs it),
//!   and the verifier challenge (`lower::challenge::CHALLENGE`: halo2-base
//!   only).
//! * Postconditions: every rank's argsort index, every rank's sorted value
//!   when one is live, and every carried payload's entry per rank: NumPy's
//!   stable sort, as the routed network (`routed`) gives them.
//! * Emission: each element is packed once, `K_j = a_j·2^b + j`
//!   (`IR::PackKey`, `b` = bits of `n - 1`). The prover supplies the sorted
//!   keys `y_r` (`HintKind::SortedValues` of the keys) and, for each carried
//!   array `c` (the sorted values when live, each payload), its entries in
//!   sorted order `c'_r` (`HintKind::OrderedBy`). One `IR::IdentityCheck`
//!   asserts that the tuples `(y_r, c'_r, ...)` are the tuples
//!   `(K_j, c_j, ...)` as multisets; the `y_r` are asserted strictly
//!   increasing (`IR::PackedKeyLess` under `IR::Assert`, after the check,
//!   whose multiset equality bounds them for the backend), and each live
//!   index is unpacked once (`IR::PackedKeyIndex`).
//! * Cost (`cost::grand_product_plan`, halo2-base): the packs, one hint cell
//!   per key and per carried entry, `n - 1` asserted comparisons, one
//!   unpack per live index, and in the second phase `2n` factors of the
//!   grand product (6 cells each) plus `3·carried` cells per tuple and side
//!   to compress it. Against the routed network it drops the `W(n) = O(n
//!   log n)` switches (16 cells each, plus 12 per carried array).
//!
//! Soundness (exact except for the check's error; the relaxed hint rule is
//! not needed, every hint is unique):
//! 1. *The hinted tuples are the input tuples, reordered.* The multiset
//!    check holds except with probability `n·max(1, w - 1) / |F|` for `w =
//!    1 + carried` (`Identity::degree`, `prove/halo2_base/identities.rs`),
//!    so the list `(y_r, c'_r)` is a permutation of `(K_j, c_j)`: tuples
//!    move as wholes, so `c'_r` is the entry of the element whose key is
//!    `y_r`.
//! 2. *Sortedness is checked.* The `y_r` are asserted strictly increasing.
//!    Packed keys are distinct (their low `b` bits are the distinct indices
//!    `j < 2^b`) and their integer order is the order of `(a_j, j)`
//!    (`IR::KeyLess`); by 1 every `y_r` is one of them (in the packed-key
//!    range, so the asserted gap check is exact), and exactly one
//!    arrangement passes: ascending.
//! 3. *Every hint is unique.* By 1 and 2, `y_r` is the `r`-th smallest key
//!    and `c'_r` the entry of the element of rank `r` in NumPy's stable
//!    order (the index tie-break is part of the key). The unpacked index is
//!    pinned by `IR::PackedKeyIndex`.
//!
//! Completeness: the honest hints (the kernel's stable argsort) satisfy the
//! multiset identity whatever the challenges, and their keys increase.

use crate::builder::IRBuilder;
use crate::ir_defs::{HintKind, Identity};
use crate::types::Value;

use super::emit::{Elements, Payload};

/// The sorted values (when `with_values`, else none) and the argsort
/// indices of `elements`, every rank, from hinted sorted keys checked by a
/// grand product; each payload's slots are replaced by its entries in
/// sorted order.
pub(super) fn sort_by_grand_product(
    b: &mut IRBuilder,
    elements: &Elements,
    with_values: bool,
    payloads: &mut [Payload],
) -> (Vec<Value>, Vec<Value>) {
    let n = elements.values.len();
    let bits = elements.index_bits();
    let keys: Vec<Value> = elements
        .values
        .iter()
        .enumerate()
        .map(|(j, v)| {
            let index = b.ir_constant_int(j as i64);
            b.ir_pack_key(v, &index, bits)
        })
        .collect();
    let lanes = |b: &mut IRBuilder, kind: HintKind, operands: &[Value]| -> Vec<Value> {
        (0..n)
            .map(|r| {
                let rank = b.ir_constant_int(r as i64);
                b.ir_hint(kind, operands, &rank)
            })
            .collect()
    };
    let sorted_keys = lanes(b, HintKind::SortedValues, &keys);
    // The carried arrays: the values when live, then each payload.
    let mut carried: Vec<&[Value]> = Vec::new();
    if with_values {
        carried.push(elements.values);
    }
    carried.extend(payloads.iter().map(|p| p.slots.as_slice()));
    let sorted_carried: Vec<Vec<Value>> = carried
        .iter()
        .map(|entries| {
            let operands: Vec<Value> = elements.values.iter().chain(entries.iter()).cloned().collect();
            lanes(b, HintKind::OrderedBy, &operands)
        })
        .collect();
    let width = 1 + carried.len();
    let tuple = |key: &Value, entries: Vec<Value>| std::iter::once(key.clone()).chain(entries);
    let left = (0..n).flat_map(|j| tuple(&keys[j], carried.iter().map(|c| c[j].clone()).collect()));
    let right = (0..n).flat_map(|r| tuple(&sorted_keys[r], sorted_carried.iter().map(|c| c[r].clone()).collect()));
    let operands: Vec<Value> = left.chain(right).collect();
    b.ir_identity_check(Identity::MultisetEqual { len: n as u32, width: width as u32 }, &operands);
    for pair in sorted_keys.windows(2) {
        let increasing = b.ir_packed_key_less(&pair[0], &pair[1], bits);
        b.ir_assert(&increasing);
    }
    let indices: Vec<Value> = sorted_keys.iter().map(|key| b.ir_packed_key_index(key, bits)).collect();
    let mut sorted_carried = sorted_carried.into_iter();
    let values = if with_values { sorted_carried.next().expect("the values") } else { Vec::new() };
    for (payload, sorted) in payloads.iter_mut().zip(sorted_carried) {
        payload.slots = sorted;
    }
    (values, indices)
}
