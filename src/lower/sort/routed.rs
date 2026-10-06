//! The stable sort by a Waksman permutation network whose switches the
//! prover sets (`Plan::Routed`, construction [`super::RoutedPermutation`]).
//!
//! Construction.
//! * Preconditions: the sort's elements (int64 values or Q32 codes; the
//!   backend checks each into the signed 64-bit window when it packs it).
//! * Postconditions: every rank's argsort index, every rank's sorted value
//!   when one is live, and every carried payload's entry per rank: NumPy's
//!   stable sort (`np.sort` / `np.argsort(kind="stable")`, `b[np.argsort(a)]`).
//! * Emission: each element is packed once as `K_j = a_j·2^b + j`
//!   (`IR::PackKey`, `b` = bits of `n - 1`). The prover supplies one bit per
//!   switch of the Waksman network on `n` wires (`HintKind::RoutingBits`,
//!   `waksman::switch_count(n)` lanes: 573 for 100 elements), each asserted
//!   a bit (`IR::AssertBit`). A switch moves its two wires, the key and the
//!   value and payload entries carried with it, by two `IR::SelectBitI` /
//!   `IR::SelectBitF` on its bit. The routed keys are asserted to increase
//!   strictly, `K'_r < K'_{r+1}` (`IR::PackedKeyLess` under `IR::Assert`),
//!   and each live index is unpacked once (`IR::PackedKeyIndex`).
//! * Cost (`cost::routed_plan`, halo2-base only): `n` packs, `W(n)`
//!   switches (a bit gate holding the hint and two selects of the key, plus
//!   two selects per carried array), `n - 1` asserted comparisons (one range
//!   check of the gap each), one unpack per live index. Batcher's merge
//!   exchange needs `O(n log^2 n)` comparisons (1,077 for 100 keys); this
//!   needs `W(n) = O(n log n)` switches of selects and only `n - 1`
//!   comparisons. Not offered on Noir: there the permutation check reads
//!   each hinted key from native memory (`PermutationCheck`), linear in `n`
//!   and cheaper than `W(n)` switches.
//!
//! Soundness (the relaxed hint rule, `IR::Hint`): every output, and every
//! value an output or a check depends on, is determined by the elements;
//! only the switch bits may vary, and no accepted variation changes any of
//! those values.
//! 1. *The routed wires are a permutation of the input wires.* Each bit is
//!    asserted to be 0 or 1, so each switch outputs its two input wires
//!    either straight or exchanged: a bijection on its two wires. The
//!    network is a fixed composition of such switches, so under every
//!    setting the routed wires are the input wires reordered, each exactly
//!    once (`waksman`: "every setting realizes a permutation", tested
//!    exhaustively). A wire is moved as a whole (key, value, payload
//!    entries on the same bit), so routed slot `r` holds the key, value and
//!    payload entries of one and the same element.
//! 2. *Sortedness is checked.* The routed keys are asserted strictly
//!    increasing. Packed keys are distinct (their low `b` bits are the
//!    distinct indices `j < 2^b`) and their integer order is the
//!    lexicographic order of `(a_j, j)` (`IR::KeyLess`), so exactly one
//!    arrangement of the `n` keys passes: ascending. Wrong orders, equal
//!    neighbours and (by 1) duplicated or dropped keys are rejected.
//! 3. *The outputs are unique.* By 1 and 2 routed slot `r` holds the
//!    element whose key `(a_j, j)` is the `r`-th smallest: for the sorted
//!    values the `r`-th smallest value (the multiset of values is the
//!    input's, in ascending order), for the argsort index and the payload
//!    entries those of the element of rank `r` in NumPy's stable order (the
//!    index tie-break is part of the key, so equal values are in their
//!    original order and no other arrangement is accepted). The unpacked
//!    index is pinned by `IR::PackedKeyIndex` (its decomposition with the
//!    value in the window and the index in `[0, 2^b)` is unique).
//! 4. *The switch bits themselves are not unique.* The network can realize
//!    one permutation by several settings (for `n = 4`, 5 switches give 32
//!    settings for 24 permutations). By 2 every accepted setting realizes
//!    the same permutation (the one ascending arrangement of distinct
//!    keys), so by 1 every wire value after the last switch, hence every
//!    output and every checked value, is the same under all of them. The
//!    intermediate wires (between switches) may differ between accepted
//!    settings; nothing reads them but later switches.
//!
//! Completeness: the looping algorithm (`waksman::route`, `kernel::
//! hint_values`) gives bits that route element `j` to its rank, under which
//! every check holds. The tamper tests (`tests.rs`, `prove/halo2_base/
//! synthesizer_tests.rs`) enumerate every setting of small networks, with
//! all keys equal and with ties: the accepted ones give NumPy's outputs and
//! are more than one; a non-bit (2, -1, 1/2: the last makes both outputs of
//! a switch `(x + y) / 2`, a duplicate) is rejected.

use crate::builder::IRBuilder;
use crate::ir_defs::HintKind;
use crate::types::{NumberType, Value};

use super::emit::{Elements, Payload};
use super::waksman;

/// One wire of the network: a packed key and the entries carried with it.
#[derive(Clone)]
struct Wire {
    key: Value,
    value: Option<Value>,
    carried: Vec<Value>,
}

/// The sorted values (when `with_values`, else none) and the argsort
/// indices of `elements`, every rank, by a routed Waksman network; each
/// payload's slots are replaced by its entries in sorted order. Outputs no
/// lane uses are removed by dead-code elimination (the routed keys are not:
/// their order is asserted).
pub(super) fn sort_by_routing(
    b: &mut IRBuilder,
    elements: &Elements,
    with_values: bool,
    payloads: &mut [Payload],
) -> (Vec<Value>, Vec<Value>) {
    let bits = elements.index_bits();
    let wires: Vec<Wire> = elements
        .values
        .iter()
        .enumerate()
        .map(|(j, v)| {
            let index = b.ir_constant_int(j as i64);
            Wire {
                key: b.ir_pack_key(v, &index, bits),
                value: with_values.then(|| v.clone()),
                carried: payloads.iter().map(|p| p.slots[j].clone()).collect(),
            }
        })
        .collect();
    let carried_types: Vec<NumberType> = payloads.iter().map(|p| p.dtype).collect();
    let mut switch = 0usize;
    let routed = waksman::apply(wires, &mut |x: Wire, y: Wire| {
        let crossed = switch_bit(b, elements, switch);
        switch += 1;
        let top = pick(b, &crossed, &y, &x, elements.dtype, &carried_types);
        let bottom = pick(b, &crossed, &x, &y, elements.dtype, &carried_types);
        (top, bottom)
    });
    debug_assert_eq!(switch, waksman::switch_count(elements.values.len()));
    for pair in routed.windows(2) {
        let increasing = b.ir_packed_key_less(&pair[0].key, &pair[1].key, bits);
        b.ir_assert(&increasing);
    }
    let indices: Vec<Value> = routed.iter().map(|w| b.ir_packed_key_index(&w.key, bits)).collect();
    let values: Vec<Value> = routed.iter().filter_map(|w| w.value.clone()).collect();
    for (k, payload) in payloads.iter_mut().enumerate() {
        payload.slots = routed.iter().map(|w| w.carried[k].clone()).collect();
    }
    (values, indices)
}

/// Switch `t`'s setting (1 = crossed): lane `t` of the routing hint over the
/// elements, asserted a bit.
fn switch_bit(b: &mut IRBuilder, elements: &Elements, t: usize) -> Value {
    let lane = b.ir_constant_int(t as i64);
    let bit = b.ir_hint(HintKind::RoutingBits, elements.values, &lane);
    b.ir_assert_bit(&bit);
    bit
}

/// `crossed ? a : other`, wire by wire.
fn pick(b: &mut IRBuilder, crossed: &Value, a: &Wire, other: &Wire, dtype: NumberType, carried: &[NumberType]) -> Wire {
    let select = |b: &mut IRBuilder, dtype: NumberType, t: &Value, f: &Value| match dtype {
        NumberType::Float => b.ir_select_bit_f(crossed, t, f),
        _ => b.ir_select_bit_i(crossed, t, f),
    };
    Wire {
        key: b.ir_select_bit_i(crossed, &a.key, &other.key),
        value: match (&a.value, &other.value) {
            (Some(t), Some(f)) => Some(select(b, dtype, t, f)),
            _ => None,
        },
        carried: a
            .carried
            .iter()
            .zip(&other.carried)
            .zip(carried)
            .map(|((t, f), &d)| select(b, d, t, f))
            .collect(),
    }
}
