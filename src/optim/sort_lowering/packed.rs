//! The stable argsort by a sorting network that exchanges one packed key
//! per slot.
//!
//! Construction (`Network::PackedMergeExchange`).
//! * Preconditions: the sort's elements (values in the signed 64-bit
//!   window, which the backends check when packing); only argsort indices
//!   are live (no sorted value).
//! * Postconditions: every rank's argsort index, NumPy's stable order.
//! * Emission: each element `a_j` is packed once as `K_j = a_j·2^b + j`
//!   (`IR::PackKey`, `b` = bits of `n - 1`); every comparator of Batcher's
//!   merge exchange compares two packed keys (`IR::PackedKeyLess`) and
//!   exchanges them with two selects; each output slot's index is unpacked
//!   once (`IR::PackedKeyIndex`). Against the `(value, index)` network
//!   (`Network::MergeExchangeTieBreak`) a comparator repacks no key and
//!   moves one cell instead of two.
//! * Cost: `n` packs, `|cone|` comparators of one comparison and two
//!   selects, one unpack per live index (`cost::UnitCosts`).
//!
//! Soundness and completeness. No hint is involved in the order: every
//! selection's condition is a comparison the backend decides exactly, so
//! every slot holds, in every accepted witness, a permutation of the
//! packed keys, sorted by the network (Batcher's network sorts any input;
//! see `cost::merge_exchange_pairs`). Packed keys are distinct (their low
//! bits are the distinct indices) and their integer order is the
//! lexicographic order of `(a_j, j)` (`IR::KeyLess`), whose ascending order
//! is NumPy's stable argsort. The unpacking is pinned by its backend
//! (`IR::PackedKeyIndex`: the decomposition with the value in the window
//! and the index in `[0, 2^b)` is unique), so slot `r` yields lane `r` of
//! the stable argsort and nothing else.

use crate::builder::IRBuilder;
use crate::types::Value;

use super::cost::merge_exchange_pairs;
use super::emit::Elements;

/// The stable argsort of `elements`, every rank, by Batcher's merge
/// exchange on packed keys. Comparators and unpacks no live output needs
/// are removed by dead-code elimination.
pub(super) fn argsort_by_packed_keys(b: &mut IRBuilder, elements: &Elements) -> Vec<Value> {
    let n = elements.values.len();
    let bits = elements.index_bits();
    let mut keys: Vec<Value> = elements
        .values
        .iter()
        .enumerate()
        .map(|(j, v)| {
            let index = b.ir_constant_int(j as i64);
            b.ir_pack_key(v, &index, bits)
        })
        .collect();
    for (i, j) in merge_exchange_pairs(n) {
        let swap = b.ir_packed_key_less(&keys[j], &keys[i], bits);
        let low = b.ir_select_i(&swap, &keys[j], &keys[i]);
        let high = b.ir_select_i(&swap, &keys[i], &keys[j]);
        keys[i] = low;
        keys[j] = high;
    }
    keys.iter().map(|key| b.ir_packed_key_index(key, bits)).collect()
}
