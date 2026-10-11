//! NumPy behaviours that several cases need: indexing, reductions,
//! `isclose`, sorting, membership, the bounding-box crop and int64 bit
//! patterns. Each function follows NumPy's semantics and fails where NumPy
//! raises. Cairo arrays are append-only, so functions that reorder build new
//! arrays (merge sort).
use core::cmp::{max, min};
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::q32::{Q32, Q32Trait, ZERO};

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

/// A Python/NumPy index into an axis of length `len`: negative indices count
/// from the end; anything outside `[-len, len)` fails like NumPy's IndexError.
pub fn np_index(i: i64, len: u32) -> u32 {
    let n: i64 = len.into();
    assert!(-n <= i && i < n, "index out of range");
    let wrapped = if i < 0 {
        i + n
    } else {
        i
    };
    wrapped.try_into().unwrap()
}

/// `np.insert`'s position on an axis of length `len`: anything in
/// `[-len, len]`, negative positions counting from the end; NumPy raises
/// `IndexError` outside that range.
pub fn insert_position(pos: i64, len: u32) -> u32 {
    let n: i64 = len.into();
    assert!(-n <= pos && pos <= n, "insert position out of range");
    let wrapped = if pos < 0 {
        pos + n
    } else {
        pos
    };
    wrapped.try_into().unwrap()
}

/// `c = np.empty_like(permutation); c[permutation] = np.arange(n)`, the
/// inverse permutation. An entry that no index writes would keep
/// `np.empty_like`'s uninitialized value, so `permutation` must hit every
/// position, i.e. be a permutation (negative entries count from the end).
pub fn inverse_permutation(permutation: Span<i64>) -> Array<u32> {
    let n = permutation.len();
    // `i + 1` at key `permutation[i]`; the default 0 marks an unwritten entry.
    let mut written: Felt252Dict<u32> = Default::default();
    for i in 0..n {
        written.insert(np_index(*permutation.at(i), n).into(), i + 1);
    }
    let mut c = array![];
    for j in 0..n {
        let entry = written.get(j.into());
        assert!(entry != 0, "permutation leaves an entry of c uninitialized");
        c.append(entry - 1);
    }
    c
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// `np.argmax`: the index of the first occurrence of the maximum. Reading
/// `values[0]` fails on an empty span, which NumPy refuses too.
pub fn argmax<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(values: Span<T>) -> u32 {
    let mut best = 0;
    let mut largest = *values.at(0);
    for i in 1..values.len() {
        let v = *values.at(i);
        if v > largest {
            best = i;
            largest = v;
        }
    }
    best
}

/// `np.argmin`: the index of the first occurrence of the minimum. Reading
/// `values[0]` fails on an empty span, which NumPy refuses too.
pub fn argmin<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(values: Span<T>) -> u32 {
    let mut best = 0;
    let mut smallest = *values.at(0);
    for i in 1..values.len() {
        let v = *values.at(i);
        if v < smallest {
            best = i;
            smallest = v;
        }
    }
    best
}

/// `np.mean` of Q32 values: the exact sum divided by the count. NumPy gives
/// nan for an empty array; that division by zero fails here.
pub fn mean(values: Span<Q32>) -> Q32 {
    let mut sum = ZERO;
    for v in values {
        sum = sum + *v;
    }
    sum.div_int(values.len().into())
}

/// `np.std` (population, `ddof=0`) of values whose mean is `mean`:
/// `sqrt(sum((v - mean)^2) / n)`. The caller passes `mean(values)`, so a
/// kernel that also uses the mean computes it once.
pub fn std(values: Span<Q32>, mean: Q32) -> Q32 {
    let mut squares = ZERO;
    for v in values {
        let d = *v - mean;
        squares = squares + d * d;
    }
    squares.div_int(values.len().into()).sqrt()
}

/// `np.abs` of an int64.
pub fn abs(v: i64) -> i64 {
    if v < 0 {
        -v
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// Comparison and membership
// ---------------------------------------------------------------------------

/// `np.isclose`'s default `rtol = 1e-5`, quantized: `round(1e-5 * 2^32)`.
pub const DEFAULT_RTOL: Q32 = Q32 { bits: 42950 };
/// `np.isclose`'s default `atol = 1e-8`, quantized: `round(1e-8 * 2^32)`.
pub const DEFAULT_ATOL: Q32 = Q32 { bits: 43 };

/// `np.isclose(a, b, rtol, atol)`: `|a - b| <= atol + rtol * |b|` (not
/// symmetric in `a` and `b`). Q32 values are never nan, so `equal_nan` has no
/// effect. Inlined: as a call it costs more steps than the comparison itself.
#[inline(always)]
pub fn isclose(a: Q32, b: Q32, rtol: Q32, atol: Q32) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

/// `value in values`: `np.in1d` / `np.isin` for one element.
pub fn contains<T, +Copy<T>, +Drop<T>, +PartialEq<T>>(values: Span<T>, value: T) -> bool {
    for v in values {
        if *v == value {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// `np.sort`: a stable merge sort.
pub fn sort<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(values: Span<T>) -> Array<T> {
    if values.len() <= 1 {
        let mut out = array![];
        for v in values {
            out.append(*v);
        }
        return out;
    }
    let mid = values.len() / 2;
    let left = sort(values.slice(0, mid));
    let right = sort(values.slice(mid, values.len() - mid));
    let (left, right) = (left.span(), right.span());
    let mut out = array![];
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if *right.at(j) < *left.at(i) {
            out.append(*right.at(j));
            j += 1;
        } else {
            out.append(*left.at(i));
            i += 1;
        }
    }
    out.append_span(left.slice(i, left.len() - i));
    out.append_span(right.slice(j, right.len() - j));
    out
}

/// `np.argsort`: the indices that sort `values`, ties in index order (a
/// stable merge sort of the indices).
pub fn argsort<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(values: Span<T>) -> Array<u32> {
    let mut indices = array![];
    for i in 0..values.len() {
        indices.append(i);
    }
    argsort_indices(values, indices.span())
}

fn argsort_indices<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(
    values: Span<T>, indices: Span<u32>,
) -> Array<u32> {
    if indices.len() <= 1 {
        let mut out = array![];
        out.append_span(indices);
        return out;
    }
    let mid = indices.len() / 2;
    let left = argsort_indices(values, indices.slice(0, mid));
    let right = argsort_indices(values, indices.slice(mid, indices.len() - mid));
    let (left, right) = (left.span(), right.span());
    let mut out = array![];
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if *values.at(*right.at(j)) < *values.at(*left.at(i)) {
            out.append(*right.at(j));
            j += 1;
        } else {
            out.append(*left.at(i));
            i += 1;
        }
    }
    out.append_span(left.slice(i, left.len() - i));
    out.append_span(right.slice(j, right.len() - j));
    out
}

/// `np.argsort(a, axis=0)` of a row-major matrix with `cols` columns, by
/// column: entry `j` holds the row indices that sort column `j`, ties in row
/// order, so `result[j][i] == np.argsort(a, axis=0)[i, j]`.
pub fn argsort_axis0<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(
    a: Span<T>, cols: u32,
) -> Array<Span<u32>> {
    let rows = a.len() / cols;
    let mut orders = array![];
    for j in 0..cols {
        let mut column = array![];
        for i in 0..rows {
            column.append(*a.at(i * cols + j));
        }
        orders.append(argsort(column.span()).span());
    }
    orders
}

// ---------------------------------------------------------------------------
// Building arrays
// ---------------------------------------------------------------------------

/// Appends `count` copies of `value`: the fill of `np.zeros`, `np.full` and
/// `np.pad`.
pub fn append_repeated<T, +Copy<T>, +Drop<T>>(ref out: Array<T>, value: T, count: u32) {
    for _ in 0..count {
        out.append(value);
    }
}

// ---------------------------------------------------------------------------
// Bounding-box crop
// ---------------------------------------------------------------------------

/// A block of cells of a matrix: rows `top..=bottom`, columns `left..=right`.
#[derive(Copy, Drop, Debug, PartialEq)]
pub struct Window {
    pub top: u32,
    pub bottom: u32,
    pub left: u32,
    pub right: u32,
}

/// The smallest window of a row-major matrix with `cols` columns that holds
/// every cell where `hit` is true, or `None` if no cell does: the per-axis min
/// and max of `np.argwhere(mask)`, equally the ranges that
/// `np.flatnonzero(mask.sum(axis=1))` and `np.flatnonzero(mask.sum(axis=0))`
/// span.
///
/// `hit` is a closure from a cell to `bool`; the `Into` bound states that
/// without the experimental `[Output: bool]` constraint syntax.
pub fn bounding_box<
    T, +Copy<T>, +Drop<T>, F, +Drop<F>, impl Hit: core::ops::Fn<F, (T,)>, +Into<Hit::Output, bool>,
>(
    a: Span<T>, cols: u32, hit: F,
) -> Option<Window> {
    let mut window: Option<Window> = None;
    // The row and column of `v`.
    let (mut i, mut j) = (0, 0);
    for v in a {
        if hit(*v).into() {
            window = Some(extend(window, i, j));
        }
        j += 1;
        if j == cols {
            i += 1;
            j = 0;
        }
    }
    window
}

/// `window` grown to hold cell `(i, j)`; rows are scanned in order, so `i` is
/// its new last row.
fn extend(window: Option<Window>, i: u32, j: u32) -> Window {
    match window {
        None => Window { top: i, bottom: i, left: j, right: j },
        Some(w) => Window { top: w.top, bottom: i, left: min(w.left, j), right: max(w.right, j) },
    }
}

/// `a[top:bottom+1, left:right+1]` of a row-major matrix with `cols` columns:
/// its data-dependent shape `[height, width]` and its values, row-major.
pub fn crop<T, +Copy<T>, +Drop<T>>(a: Span<T>, cols: u32, w: Window) -> (Array<i64>, Array<T>) {
    let (height, width) = (w.bottom - w.top + 1, w.right - w.left + 1);
    let mut values = array![];
    for i in w.top..w.bottom + 1 {
        values.append_span(a.slice(i * cols + w.left, width));
    }
    (array![height.into(), width.into()], values)
}

// ---------------------------------------------------------------------------
// int64 bit patterns
// ---------------------------------------------------------------------------

/// The int64 two's-complement bit pattern of `v` as a `u64`: `v` itself, or
/// `v + 2^64` if `v < 0`. NumPy's bitwise operators (`&`, `|`, `^`) act on
/// that pattern; Cairo has no bitwise operators on signed integers.
pub fn twos_complement(v: i64) -> u64 {
    let wide: i128 = v.into();
    if wide < 0 {
        (wide + 0x10000000000000000).try_into().unwrap()
    } else {
        wide.try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::q32::{Q32, Q32Trait, ZERO};
    use super::{
        DEFAULT_ATOL, DEFAULT_RTOL, Window, abs, append_repeated, argmax, argmin, argsort,
        argsort_axis0, bounding_box, contains, crop, insert_position, inverse_permutation, isclose,
        mean, np_index, sort, std, twos_complement,
    };

    fn q(n: i64) -> Q32 {
        Q32Trait::from_int(n)
    }

    #[test]
    fn sorts() {
        assert!(sort(array![3_i64, -1, 2, -1, 0].span()) == array![-1, -1, 0, 2, 3]);
        assert!(argsort(array![3_i64, -1, 2, -1, 0].span()) == array![1, 3, 4, 2, 0]);
        // Columns [2, 1, 3] and [0, 0, -1].
        let orders = argsort_axis0(array![q(2), q(0), q(1), q(0), q(3), q(-1)].span(), 2);
        assert!(*orders.at(0) == array![1, 0, 2].span());
        assert!(*orders.at(1) == array![2, 0, 1].span());
    }

    #[test]
    fn indexes_like_numpy() {
        assert!(np_index(-1, 5) == 4);
        assert!(np_index(2, 5) == 2);
        assert!(insert_position(5, 5) == 5);
        assert!(insert_position(-5, 5) == 0);
        assert!(insert_position(-1, 5) == 4);
        assert!(inverse_permutation(array![2, 0, -2].span()) == array![1, 2, 0]);
    }

    #[test]
    #[should_panic(expected: "insert position out of range")]
    fn insert_past_the_end_fails() {
        insert_position(6, 5);
    }

    #[test]
    #[should_panic(expected: "permutation leaves an entry of c uninitialized")]
    fn inverse_of_a_non_permutation_fails() {
        inverse_permutation(array![0, 0, 1].span());
    }

    #[test]
    fn arg_extrema_take_the_first_occurrence() {
        let values = array![q(1), q(3), q(-2), q(3), q(-2)].span();
        assert!(argmax(values) == 1);
        assert!(argmin(values) == 2);
    }

    #[test]
    #[should_panic(expected: ('Index out of bounds',))]
    fn argmax_of_nothing_fails() {
        let empty: Array<Q32> = array![];
        argmax(empty.span());
    }

    #[test]
    fn mean_and_std() {
        let values = array![q(2), q(4), q(4), q(4), q(5), q(5), q(7), q(9)].span();
        let m = mean(values);
        assert!(m == q(5));
        assert!(std(values, m) == q(2));
        // The division rounds toward negative infinity.
        assert!(mean(array![q(-1), ZERO, ZERO].span()) == Q32 { bits: -1431655766 });
        assert!(abs(-7) == 7 && abs(7) == 7);
    }

    #[test]
    fn isclose_is_relative_to_the_second_argument() {
        let b = q(100000);
        assert!(isclose(b + q(1), b, DEFAULT_RTOL, DEFAULT_ATOL));
        assert!(!isclose(b + q(2), b, DEFAULT_RTOL, DEFAULT_ATOL));
        assert!(isclose(Q32 { bits: 43 }, ZERO, DEFAULT_RTOL, DEFAULT_ATOL));
        assert!(!isclose(Q32 { bits: 43 }, ZERO, DEFAULT_RTOL, ZERO));
    }

    #[test]
    fn membership() {
        let set = array![3_i64, -1, 2].span();
        assert!(contains(set, -1));
        assert!(!contains(set, 0));
    }

    #[test]
    fn appends_copies() {
        let mut out = array![1_i64];
        append_repeated(ref out, 0, 3);
        assert!(out == array![1, 0, 0, 0]);
    }

    #[test]
    fn crops_to_the_bounding_box() {
        let a = array![0_i64, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0].span();
        let window = bounding_box(a, 4, |v: i64| v != 0).unwrap();
        assert!(window == Window { top: 1, bottom: 2, left: 1, right: 3 });
        let (shape, values) = crop(a, 4, window);
        assert!(shape == array![2, 3]);
        assert!(values == array![1, 0, 0, 0, 0, 2]);
        assert!(bounding_box(a, 4, |v: i64| v > 5).is_none());
    }

    #[test]
    fn twos_complement_bit_patterns() {
        assert!(twos_complement(5) == 5);
        assert!(twos_complement(-1) == 0xffffffffffffffff);
        assert!(twos_complement(-2) & 1 == 0);
    }
}

/// Fused row-major contractions, matching zkvm_common/kernels/src/np.rs.
pub fn matmul(a: Span<Q32>, b: Span<Q32>, rows: u32, inner: u32, cols: u32) -> Array<Q32> {
    assert!(a.len() == rows * inner && b.len() == inner * cols, "matmul shape mismatch");
    let mut out = array![];
    for i in 0..rows {
        for j in 0..cols {
            let mut raw: i128 = 0;
            for k in 0..inner { raw += crate::q32::raw_product(*a.at(i * inner + k), *b.at(k * cols + j)); };
            out.append(crate::q32::from_raw_product(raw));
        };
    };
    out
}
pub fn matvec(a: Span<Q32>, b: Span<Q32>, rows: u32, cols: u32) -> Array<Q32> {
    matmul(a, b, rows, cols, 1)
}
pub fn vecmat(a: Span<Q32>, b: Span<Q32>, rows: u32, cols: u32) -> Array<Q32> {
    matmul(a, b, 1, rows, cols)
}
pub fn inner(a: Span<Q32>, b: Span<Q32>) -> Q32 {
    assert!(a.len() == b.len(), "inner shape mismatch");
    let mut raw: i128 = 0;
    for i in 0..a.len() { raw += crate::q32::raw_product(*a.at(i), *b.at(i)); };
    crate::q32::from_raw_product(raw)
}

#[cfg(test)]
mod contraction_tests {
    use super::{matmul, matvec, vecmat, inner};
    use crate::q32::{Q32, Q32Trait, ZERO, ONE, HALF};

    #[test]
    fn contractions_rescale_once() {
        let a = array![HALF, HALF].span();
        let b = array![Q32 { bits: 1 }, Q32 { bits: 1 }].span();
        assert!(inner(a, b).bits == 1);
        assert!(inner(array![-HALF, -HALF].span(), b).bits == -1);
        assert!(matvec(a, b, 1, 2) == array![Q32 { bits: 1 }]);
        assert!(vecmat(a, b, 2, 1) == array![Q32 { bits: 1 }]);
    }
    #[test]
    fn rectangular_and_empty_products() {
        let a = array![ONE, Q32Trait::from_int(2), Q32Trait::from_int(3)].span();
        let b = array![ONE, ZERO, ZERO, ONE, ONE, ONE].span();
        assert!(matmul(a, b, 1, 3, 2) == array![Q32Trait::from_int(4), Q32Trait::from_int(5)]);
        assert!(matmul(array![].span(), array![].span(), 2, 0, 3) == array![ZERO, ZERO, ZERO, ZERO, ZERO, ZERO]);
        assert!(inner(array![].span(), array![].span()) == ZERO);
    }
    #[test]
    #[should_panic]
    fn rejects_mismatched_shapes() {
        let _ = matmul(array![ONE].span(), array![ONE].span(), 2, 1, 1);
    }
}
