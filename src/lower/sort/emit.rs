//! The IR of each lowering of a sort's outputs (see `super`).
//!
//! Terms. The *key* of element `j` is the pair `(a_j, j)`, ordered
//! lexicographically. Keys are distinct, and their ascending order is
//! NumPy's stable sort: equal values in original order. The output of rank
//! `r` is the element whose key has exactly `r` keys below it: its value is
//! the `r`-th smallest value, its index lane `r` of the stable argsort.
//!
//! Comparisons are the IR's signed comparisons, which the proving backends
//! decide exactly on the signed 64-bit window (and reject operands outside
//! it), the domain of Zinnia's int64 values and Q32 floats. Counts are sums
//! of comparison bits, so they are integers in `[0, n]`.

use crate::builder::IRBuilder;
use crate::ir_defs::HintKind;
use crate::types::{NumberType, Value};

use super::cost::Network;

/// The elements of one sort and their number type (it picks the float or
/// integer form of each comparison; backends decide both the same way).
pub(super) struct Elements<'a> {
    pub values: &'a [Value],
    pub dtype: NumberType,
    /// Bounds every element's value (or float code) lies in on every
    /// accepted witness, when the layer established them (the array claim
    /// `ElementsWithin`, a precondition of the plan), else `None`.
    pub range: Option<(i64, i64)>,
}

impl Elements<'_> {
    fn len(&self) -> usize {
        self.values.len()
    }

    /// Bound `v`, a value the plan takes to be an element (a hinted sorted
    /// value, a read at a hinted index), into the elements' range, when one
    /// is established: one narrow check `0 <= v - lo <= hi - lo`, after
    /// which every comparison `v` takes part in is narrow on both backends
    /// (halo2-base sizes it by the proven ranges, Noir by the interval the
    /// check deposits). Complete: the value the constraints admit is an
    /// element, inside the range on every accepted witness. Sound: it only
    /// adds a constraint.
    pub(super) fn bound(&self, b: &mut IRBuilder, v: &Value) {
        let Some((lo, hi)) = self.range else { return };
        let code = Value::Integer(crate::types::ScalarValue::new(None, v.stmt_id()));
        let shifted = if lo == 0 {
            code
        } else {
            let lo = b.ir_constant_int(lo);
            b.ir_sub_i(&code, &lo)
        };
        let size = b.ir_constant_int(hi - lo + 1);
        b.ir_assert_u128_lt(&shifted, &size);
    }

    /// Bits that hold every index `0..n` (at least one).
    pub(super) fn index_bits(&self) -> u32 {
        (usize::BITS - self.len().saturating_sub(1).leading_zeros()).max(1)
    }

    /// `[(v1, i1) < (v2, i2)]` lexicographically (`IR::KeyLess`); every
    /// index the lowerings pass lies in `[0, n)`: a constant, a selection
    /// among constants, or a hinted index with its bound asserted.
    fn key_less(&self, b: &mut IRBuilder, v1: &Value, i1: &Value, v2: &Value, i2: &Value) -> Value {
        b.ir_key_less(v1, i1, v2, i2, self.index_bits())
    }

    pub(super) fn less(&self, b: &mut IRBuilder, x: &Value, y: &Value) -> Value {
        match self.dtype {
            NumberType::Float => b.ir_less_than_f(x, y),
            _ => b.ir_less_than_i(x, y),
        }
    }

    pub(super) fn equal(&self, b: &mut IRBuilder, x: &Value, y: &Value) -> Value {
        match self.dtype {
            NumberType::Float => b.ir_equal_f(x, y),
            _ => b.ir_equal_i(x, y),
        }
    }

    fn select(&self, b: &mut IRBuilder, cond: &Value, t: &Value, f: &Value) -> Value {
        match self.dtype {
            NumberType::Float => b.ir_select_f(cond, t, f),
            _ => b.ir_select_i(cond, t, f),
        }
    }
}

// ── Sorting networks ────────────────────────────────────────────────────

/// An array carried through a sorting network (`b[np.argsort(a)]`): its
/// slots, exchanged with the keys.
pub(super) struct Payload {
    pub slots: Vec<Value>,
    pub dtype: NumberType,
}

impl Payload {
    /// Exchange slots `i < j` when `swap` (the key at `j` is the smaller).
    pub(super) fn exchange(&mut self, b: &mut IRBuilder, swap: &Value, i: usize, j: usize) {
        let carried = Elements { values: &self.slots, dtype: self.dtype, range: None };
        let low = carried.select(b, swap, &self.slots[j], &self.slots[i]);
        let high = carried.select(b, swap, &self.slots[i], &self.slots[j]);
        self.slots[i] = low;
        self.slots[j] = high;
    }
}

/// Sorted values and (when `network` tracks them) the original index of
/// each, by a compare-exchange network: every comparator puts the smaller
/// key first. Without a tie-break the keys are the values (equal values are
/// interchangeable, so the values are NumPy's; the adjacent-swap network
/// also keeps equal values in order, so its indices are the stable
/// argsort). With the tie-break a key is `(value, index)`, compared
/// lexicographically (`IR::KeyLess`, whose index bound holds because every
/// index cell is a selection among the constants `0..n`).
///
/// Each payload (an array as long as the elements) is exchanged with the
/// keys (`select` on the same comparison), so its slot `r` ends holding the
/// payload entry of the element at rank `r`; the network must then be
/// stable (`Network::is_stable`).
///
/// No hint: every output is a function of the elements. Comparators whose
/// outputs are not used are removed by dead-code elimination.
pub(super) fn sort_by_network(
    b: &mut IRBuilder,
    elements: &Elements,
    network: Network,
    payloads: &mut [Payload],
) -> (Vec<Value>, Vec<Value>) {
    let n = elements.len();
    let mut keys = elements.values.to_vec();
    let mut indices: Vec<Value> = if network.tracks_indices() {
        (0..n).map(|i| b.ir_constant_int(i as i64)).collect()
    } else {
        Vec::new()
    };
    for (i, j) in network.pairs(n) {
        let swap = if network.tie_break() {
            elements.key_less(b, &keys[j], &indices[j], &keys[i], &indices[i])
        } else {
            elements.less(b, &keys[j], &keys[i])
        };
        let low = elements.select(b, &swap, &keys[j], &keys[i]);
        let high = elements.select(b, &swap, &keys[i], &keys[j]);
        keys[i] = low;
        keys[j] = high;
        for payload in payloads.iter_mut() {
            payload.exchange(b, &swap, i, j);
        }
        if network.tracks_indices() {
            let low = b.ir_select_i(&swap, &indices[j], &indices[i]);
            let high = b.ir_select_i(&swap, &indices[i], &indices[j]);
            indices[i] = low;
            indices[j] = high;
        }
    }
    (keys, indices)
}

// ── Rank selectors ──────────────────────────────────────────────────────

/// The one-hot rank selectors of the stable sort, from one comparison per
/// pair of elements: `selectors[r][k] = [element k has rank r]`.
///
/// Element `k`'s rank is the number of elements before it in the stable
/// order: `j` precedes `k` when `a_j < a_k`, or `a_j == a_k` and `j < k`.
/// For `j < k` that is `not (a_k < a_j)`, for `j > k` it is `a_j < a_k`, so
/// one comparison `c_jk = [a_k < a_j]` per pair `j < k` gives every
/// `prec(j, k)`. Then `rank_k = sum_j prec(j, k)`, and
///
/// ```text
/// [rank_k == 0]     = prod_{j != k} (1 - prec(j, k))
/// [rank_k == n - 1] = prod_{j != k} prec(j, k)
/// [rank_k == 1]     = 1 - [rank_k == 0] - [rank_k == 2]      (n = 3)
/// [rank_k == r]     = [sum_j prec(j, k) == r]                (n > 3)
/// ```
///
/// (for `n = 3`, a product of two bits; for `n = 2`, the bit itself).
///
/// **Soundness.** No hint: each comparison is exact, and the stable order is
/// a strict total order on the elements, so the ranks are a permutation of
/// `0..n` and every formula above is exactly `[rank_k == r]` (rank 0 when no
/// element precedes `k`, rank `n - 1` when all do, for `n = 3` rank 1 when
/// neither). Each `selectors[r]` therefore has exactly one 1, at the element
/// of rank `r` of NumPy's stable `argsort`.
pub(super) fn rank_selectors(b: &mut IRBuilder, elements: &Elements) -> Vec<Vec<Value>> {
    let n = elements.len();
    let one = b.ir_constant_int(1);
    // before[k][j] = prec(j, k) for j != k.
    let mut before: Vec<Vec<Option<Value>>> = vec![vec![None; n]; n];
    for k in 0..n {
        for j in 0..k {
            let k_first = elements.less(b, &elements.values[k], &elements.values[j]);
            before[k][j] = Some(b.ir_sub_i(&one, &k_first));
            before[j][k] = Some(k_first);
        }
    }
    let precedes = |k: usize| -> Vec<Value> { before[k].iter().flatten().cloned().collect() };
    let mut selectors: Vec<Vec<Value>> = vec![Vec::with_capacity(n); n];
    for k in 0..n {
        let bits = precedes(k);
        let by_rank: Vec<Value> = match bits.as_slice() {
            [] => vec![one.clone()],
            [p] => vec![b.ir_sub_i(&one, p), p.clone()],
            [p, q] => {
                let (not_p, not_q) = (b.ir_sub_i(&one, p), b.ir_sub_i(&one, q));
                let first = b.ir_mul_i(&not_p, &not_q);
                let last = b.ir_mul_i(p, q);
                let ends = b.ir_add_i(&first, &last);
                vec![first, b.ir_sub_i(&one, &ends), last]
            }
            _ => {
                let rank = bits[1..].iter().fold(bits[0].clone(), |acc, bit| b.ir_add_i(&acc, bit));
                (0..n)
                    .map(|r| {
                        let r = b.ir_constant_int(r as i64);
                        b.ir_equal_i(&rank, &r)
                    })
                    .collect()
            }
        };
        for (r, selector) in by_rank.into_iter().enumerate() {
            selectors[r].push(selector);
        }
    }
    selectors
}

/// `sum_k selectors[k] * values[k]` (the entry of the selected element).
pub(super) fn select_by(b: &mut IRBuilder, selectors: &[Value], values: &[Value], dtype: NumberType) -> Value {
    let terms = selectors
        .iter()
        .zip(values)
        .map(|(bit, value)| crate::lower::gather::window::scale_by_bit(b, bit, value, dtype))
        .collect();
    crate::lower::gather::window::sum_typed(b, terms, dtype)
}

// ── Hinted keys ─────────────────────────────────────────────────────────

/// A key: a value and the index of an element.
#[derive(Clone)]
pub(super) struct Key {
    pub value: Value,
    pub index: Value,
}

/// The elements in a memory segment, for the reads through hinted indices.
#[derive(Clone, Copy)]
pub(super) struct ElementMemory {
    segment: u32,
}

impl ElementMemory {
    pub(super) fn new(b: &mut IRBuilder, elements: &Elements) -> ElementMemory {
        let segment =
            crate::helpers::segment::alloc_and_write_values(b, elements.values, elements.dtype);
        ElementMemory { segment }
    }

    /// The key of the element at the hinted index `p`: `p` is asserted to
    /// lie in `[0, n)` and the value is read at `p`, so the key is the key
    /// of an element (`a_p`, `p`), whatever the prover chose.
    fn key_at(&self, b: &mut IRBuilder, elements: &Elements, p: Value) -> Key {
        let n = b.ir_constant_int(elements.len() as i64);
        b.ir_assert_u128_lt(&p, &n);
        let value = crate::helpers::segment::read_typed(b, self.segment, &p, elements.dtype);
        elements.bound(b, &value);
        Key { value, index: p }
    }
}

/// `x < y` for keys: `x.value < y.value`, or equal values and
/// `x.index < y.index` (both indices lie in `[0, n)`: asserted, or
/// constants).
fn key_less(b: &mut IRBuilder, elements: &Elements, x: &Key, y: &Key) -> Value {
    elements.key_less(b, &x.value, &x.index, &y.value, &y.index)
}

/// The number of elements whose key lies below `key` (`above`: above it).
fn count_keys_beyond(b: &mut IRBuilder, elements: &Elements, key: &Key, above: bool) -> Value {
    let bits: Vec<Value> = (0..elements.len())
        .map(|j| {
            let element = Key {
                value: elements.values[j].clone(),
                index: b.ir_constant_int(j as i64),
            };
            if above {
                key_less(b, elements, key, &element)
            } else {
                key_less(b, elements, &element, key)
            }
        })
        .collect();
    count(b, &bits)
}

/// The sum of 0/1 values, as an integer.
pub(super) fn count(b: &mut IRBuilder, bits: &[Value]) -> Value {
    let mut total = b.ir_constant_int(0);
    for bit in bits {
        let as_int = b.ir_int_cast(bit);
        total = b.ir_add_i(&total, &as_int);
    }
    total
}

fn assert_equal_int(b: &mut IRBuilder, x: &Value, expected: i64) {
    let expected = b.ir_constant_int(expected);
    let holds = b.ir_equal_i(x, &expected);
    b.ir_assert(&holds);
}

/// The keys of ranks `first..=last`, hinted and pinned.
///
/// For each rank `r` the prover supplies `p_r` (lane `r` of the stable
/// argsort); the IR reads `v_r = a[p_r]` with `0 <= p_r < n` asserted, so
/// `(v_r, p_r)` is the key of an element. It asserts
/// 1. `(v_r, p_r) < (v_{r+1}, p_{r+1})` for consecutive ranks of the run;
/// 2. when `first > 0`: exactly `first` keys lie below the first key;
/// 3. when `last < n - 1`: exactly `n - 1 - last` keys lie above the last.
///
/// Soundness. By 1 the run's keys strictly increase, so they are
/// `last - first + 1` distinct keys between the first and the last key.
/// For a run inside the sort, 2 and 3 leave exactly
/// `n - first - (n - 1 - last) = last - first + 1` keys between the first
/// and the last key (inclusive), so the run's keys are exactly those keys,
/// in increasing order, and its `k`-th key has `first + k` keys below it.
/// At an end of the sort one count suffices: for `first = 0`, by 3 exactly
/// `last + 1` keys lie at or below the last key, and the run's `last + 1`
/// distinct keys all do, so they are those keys (symmetrically for
/// `last = n - 1` by 2); a run over every rank has `n` distinct keys, all
/// `n` keys, so it checks a permutation with `n - 1` key comparisons and no
/// count. Hence `p_r` is lane `r` of the stable argsort and `v_r` the
/// sorted value of rank `r`: the only assignment the constraints admit.
pub(super) fn pinned_run(
    b: &mut IRBuilder,
    elements: &Elements,
    memory: &ElementMemory,
    first: usize,
    last: usize,
) -> Vec<Key> {
    let n = elements.len();
    let keys: Vec<Key> = (first..=last)
        .map(|r| {
            let rank = b.ir_constant_int(r as i64);
            let p = b.ir_hint(HintKind::StableArgsort, elements.values, &rank);
            memory.key_at(b, elements, p)
        })
        .collect();
    for pair in keys.windows(2) {
        let increasing = key_less(b, elements, &pair[0], &pair[1]);
        b.ir_assert(&increasing);
    }
    if first > 0 {
        let below = count_keys_beyond(b, elements, &keys[0], false);
        assert_equal_int(b, &below, first as i64);
    }
    if last + 1 < n {
        let above = count_keys_beyond(b, elements, keys.last().expect("non-empty run"), true);
        assert_equal_int(b, &above, (n - 1 - last) as i64);
    }
    keys
}

/// The value of rank `rank` (a constant or a runtime value), pinned by
/// counting: the prover supplies `v` (lane `rank` of the sorted values) and
/// the IR asserts `count(a < v) <= rank < count(a < v) + count(a == v)`.
///
/// Soundness. Let `s_0 <= ... <= s_{n-1}` be the sorted values. For
/// `v = s_rank` both hold: elements below `s_rank` sit at sorted positions
/// below `rank`, and positions `0..=rank` hold values `<= s_rank`. For any
/// `v < s_rank`, `count(a <= v) <= count(a < s_rank) <= rank`, which breaks
/// the second; for any `v > s_rank`, positions `0..=rank` hold values below
/// `v`, so `count(a < v) >= rank + 1`, which breaks the first. So `v` is
/// pinned to `s_rank`, and a rank outside `[0, n)` admits no `v` (the
/// counts lie in `[0, n]`): no proof, as NumPy raises. The second check
/// also makes `v` equal to some element, so `v` lies in the window.
pub(super) fn order_statistic(b: &mut IRBuilder, elements: &Elements, rank: &Value) -> Value {
    let v = b.ir_hint(HintKind::SortedValues, elements.values, rank);
    elements.bound(b, &v);
    let below: Vec<Value> = elements.values.iter().map(|a| elements.less(b, a, &v)).collect();
    let ties: Vec<Value> = elements.values.iter().map(|a| elements.equal(b, a, &v)).collect();
    let below = count(b, &below);
    let ties = count(b, &ties);
    let not_above = b.ir_less_than_or_equal_i(&below, rank);
    b.ir_assert(&not_above);
    let at_or_below = b.ir_add_i(&below, &ties);
    let reached = b.ir_less_than_i(rank, &at_or_below);
    b.ir_assert(&reached);
    v
}

/// How an extreme pin shows that its value is one of the elements.
#[derive(Clone, Copy, Debug)]
pub(super) enum Membership {
    /// `prod_j g_j == 0` over the gaps `g_j = v - a_j` (`a_j - v` for the
    /// smallest) that the bound assertions check: each factor is zero
    /// exactly when `a_j - v` is, so the product is zero exactly when
    /// `prod_j (a_j - v)` is, and a backend that checks the asserted
    /// comparison by a range check of the same gap (Noir:
    /// `asserted_compare`) reuses that gap's witness as the factor.
    Product,
    /// One of the bits `[v == a_j]` (`value_first`) or `[a_j == v]` is set:
    /// their sum is non-zero. The program computes those bits anyway
    /// (`a.max(axis) == a`), so they cost nothing here.
    EqualityBits { value_first: bool },
}

/// The smallest value (`largest`: the largest), pinned without counts: the
/// prover supplies `v` (lane 0, or `n - 1`, of the sorted values) and the
/// IR asserts `v <= a_j` (`a_j <= v`) for every element and that `v` is an
/// element (`membership`).
///
/// Soundness. The asserted comparisons make `v` a lower (upper) bound of
/// the elements. A product of field elements is zero only when a factor is
/// (a field has no zero divisors); a sum of `n` bits (`n` far below the
/// field's characteristic) is non-zero only when a bit is set, and each bit
/// is an exact equality test. Either way `v` equals some element, and a
/// lower bound that is an element is the minimum: `v` is pinned. The
/// backends check an asserted comparison as one range check of its gap.
pub(super) fn extreme_value(b: &mut IRBuilder, elements: &Elements, largest: bool, membership: Membership) -> Value {
    let n = elements.len();
    let rank = b.ir_constant_int(if largest { n as i64 - 1 } else { 0 });
    let v = b.ir_hint(HintKind::SortedValues, elements.values, &rank);
    elements.bound(b, &v);
    let mut product: Option<Value> = None;
    let mut bits: Vec<Value> = Vec::new();
    for a in elements.values {
        let (low, high) = if largest { (a, &v) } else { (&v, a) };
        let bounded = match elements.dtype {
            NumberType::Float => b.ir_less_than_or_equal_f(low, high),
            _ => b.ir_less_than_or_equal_i(low, high),
        };
        b.ir_assert(&bounded);
        match membership {
            Membership::Product => {
                let difference = b.ir_sub_i(high, low);
                product = Some(match product {
                    None => difference,
                    Some(p) => b.ir_mul_i(&p, &difference),
                });
            }
            Membership::EqualityBits { value_first } => {
                let bit = if value_first { elements.equal(b, &v, a) } else { elements.equal(b, a, &v) };
                bits.push(bit);
            }
        }
    }
    let zero = b.ir_constant_int(0);
    let member = match membership {
        Membership::Product => b.ir_equal_i(&product.expect("a sort has elements"), &zero),
        Membership::EqualityBits { .. } => {
            let hits = count(b, &bits);
            b.ir_not_equal_i(&hits, &zero)
        }
    };
    b.ir_assert(&member);
    v
}

/// Lane `rank` of the stable argsort at a runtime rank: the prover supplies
/// `p`, the IR forms the key `(a_p, p)` (`p` bounded, `a_p` read) and
/// asserts that exactly `rank` keys lie below it. Keys are distinct, so
/// exactly one key has `rank` keys below it (for `rank` in `[0, n)`; none
/// otherwise, so no proof), and `p` is that key's index.
pub(super) fn argsort_at_rank(
    b: &mut IRBuilder,
    elements: &Elements,
    memory: &ElementMemory,
    rank: &Value,
) -> Value {
    let p = b.ir_hint(HintKind::StableArgsort, elements.values, rank);
    let key = memory.key_at(b, elements, p);
    let below = count_keys_beyond(b, elements, &key, false);
    let exact = b.ir_equal_i(&below, rank);
    b.ir_assert(&exact);
    key.index
}

/// Outputs of a sort in a memory segment, for reads at runtime ranks.
pub(super) struct OutputMemory {
    segment: u32,
    len: usize,
    dtype: NumberType,
}

impl OutputMemory {
    pub(super) fn new(b: &mut IRBuilder, outputs: &[Value], dtype: NumberType) -> OutputMemory {
        let segment = crate::helpers::segment::alloc_and_write_values(b, outputs, dtype);
        OutputMemory { segment, len: outputs.len(), dtype }
    }

    /// `outputs[rank]` for a runtime `rank`, with `0 <= rank < n` asserted
    /// (a rank outside has no proof, as NumPy raises).
    pub(super) fn read(&self, b: &mut IRBuilder, rank: &Value) -> Value {
        let n = b.ir_constant_int(self.len as i64);
        b.ir_assert_u128_lt(rank, &n);
        crate::helpers::segment::read_typed(b, self.segment, rank, self.dtype)
    }
}

/// The value of rank `rank` among the active elements of a runtime-length
/// array (`active[j]` is 1 when slot `j` holds an element, 0 for padding),
/// pinned by counting over the active elements only: the prover supplies `v`
/// (lane `rank` of `HintKind::MaskedSortedValues`) and the IR asserts
/// `below <= rank < below + ties` with `below = #{j active: a_j < v}` and
/// `ties = #{j active: a_j == v}`.
///
/// Soundness: [`order_statistic`]'s argument on the multiset of active
/// elements (padding contributes to neither count): `v` is pinned to their
/// value of rank `rank`, and a rank outside `[0, #active)` admits no `v`
/// (no proof, as NumPy has no such element). Emitted where the frontend
/// lowers the operation, so its asserts hold only where that code runs.
pub(crate) fn masked_order_statistic(
    b: &mut IRBuilder,
    elements: &[Value],
    active: &[Value],
    dtype: NumberType,
    rank: &Value,
) -> Value {
    let elements = Elements { values: elements, dtype, range: None };
    let mut operands = elements.values.to_vec();
    operands.extend(active.iter().cloned());
    let v = b.ir_hint(HintKind::MaskedSortedValues, &operands, rank);
    let mut below = Vec::with_capacity(active.len());
    let mut ties = Vec::with_capacity(active.len());
    for (a, flag) in elements.values.iter().zip(active) {
        let less = elements.less(b, a, &v);
        below.push(b.ir_logical_and(flag, &less));
        let equal = elements.equal(b, a, &v);
        ties.push(b.ir_logical_and(flag, &equal));
    }
    let below = count(b, &below);
    let ties = count(b, &ties);
    let not_above = b.ir_less_than_or_equal_i(&below, rank);
    b.ir_assert(&not_above);
    let at_or_below = b.ir_add_i(&below, &ties);
    let reached = b.ir_less_than_i(rank, &at_or_below);
    b.ir_assert(&reached);
    v
}
