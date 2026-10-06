//! NumPy behaviours that several cases need: `linspace`, indexing and
//! slicing, reductions over Q32 values, `isclose`, `argsort`, matrix products
//! and the bounding-box crop. Each function
//! follows NumPy's semantics and panics where NumPy raises.
//!
//! Cases reach this module through the prelude, as `np::argmax(&a)` etc.;
//! `np_index` is also in the prelude on its own.

use core::ops::{Range, RangeInclusive};

use alloc::{vec, vec::Vec};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use q32::{dot, Q32};

use crate::Shaped;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// `np.linspace(start, stop, num)`: `start + k * step` with
/// `step = (stop - start) / (num - 1)`, and the last point exactly `stop`.
/// For a step that rounds to zero, NumPy's other path `k / div * delta`.
pub fn linspace(start: Q32, stop: Q32, num: i64) -> Array1<Q32> {
    let len = usize::try_from(num).expect("Number of samples must be non-negative");
    let (div, delta) = (num - 1, stop - start);
    let mut y: Array1<Q32> = if div <= 0 {
        Array1::from_elem(len, start)
    } else {
        let step = delta / div;
        if step == Q32::ZERO {
            (0..num).map(|k| Q32::from_int(k) / div * delta + start).collect()
        } else {
            (0..num).map(|k| step * k + start).collect()
        }
    };
    if len > 1 {
        y[len - 1] = stop;
    }
    y
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

/// A Python/NumPy index into an axis of length `len`: negative indices count
/// from the end; anything outside `[-len, len)` panics like NumPy's
/// `IndexError`.
pub fn np_index(i: i64, len: usize) -> usize {
    let n = len as i64;
    assert!(-n <= i && i < n, "index out of range");
    (if i < 0 { i + n } else { i }) as usize
}

/// The indices of the Python slice `[start:stop]` (step 1) of a sequence of
/// `len` items: negative bounds count from the end, and both bounds are
/// clipped to `[0, len]`, so the range may be empty but never fails.
pub fn slice_range(start: i64, stop: i64, len: usize) -> Range<usize> {
    let n = len as i64;
    let clip = |bound: i64| (if bound < 0 { bound + n } else { bound }).clamp(0, n) as usize;
    let (start, stop) = (clip(start), clip(stop));
    start..stop.max(start)
}

/// `np.insert`'s position on an axis of length `len`: anything in
/// `[-len, len]`, negative positions counting from the end; NumPy raises
/// `IndexError` outside that range.
pub fn insert_position(pos: i64, len: usize) -> usize {
    let n = len as i64;
    assert!(-n <= pos && pos <= n, "insert position out of range");
    (if pos < 0 { pos + n } else { pos }) as usize
}

/// `c = np.empty_like(permutation); c[permutation] = np.arange(n)`, the
/// inverse permutation. An entry that no index writes would keep
/// `np.empty_like`'s uninitialized value, so `permutation` must hit every
/// position, i.e. be a permutation (negative entries count from the end).
pub fn inverse_permutation<const N: usize>(permutation: &[i64; N]) -> [usize; N] {
    let mut c = [None; N];
    for (i, &p) in permutation.iter().enumerate() {
        c[np_index(p, N)] = Some(i);
    }
    c.map(|entry| entry.expect("permutation leaves an entry of c uninitialized"))
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// `np.argmax`: the index of the first occurrence of the maximum. NumPy
/// refuses an empty array.
pub fn argmax<T: PartialOrd>(values: &[T]) -> usize {
    assert!(!values.is_empty(), "attempt to get argmax of an empty sequence");
    let mut best = 0;
    for (i, v) in values.iter().enumerate().skip(1) {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// `np.argmin`: the index of the first occurrence of the minimum. NumPy
/// refuses an empty array.
pub fn argmin<T: PartialOrd>(values: &[T]) -> usize {
    assert!(!values.is_empty(), "attempt to get argmin of an empty sequence");
    let mut best = 0;
    for (i, v) in values.iter().enumerate().skip(1) {
        if *v < values[best] {
            best = i;
        }
    }
    best
}

/// `np.mean` of Q32 values: the exact sum divided by the count. NumPy gives
/// nan for an empty array; that division by zero panics here.
pub fn mean(values: &[Q32]) -> Q32 {
    values.iter().sum::<Q32>() / values.len() as i64
}

/// `np.std` (population, `ddof=0`) of values whose mean is `mean`:
/// `sqrt(sum((v - mean)^2) / n)`. The caller passes `np::mean(values)`, so a
/// kernel that also uses the mean computes it once.
pub fn std(values: &[Q32], mean: Q32) -> Q32 {
    let variance =
        values.iter().map(|&v| (v - mean) * (v - mean)).sum::<Q32>() / values.len() as i64;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// `np.isclose`'s default `rtol = 1e-5`, quantized: `round(1e-5 * 2^32)`.
pub const DEFAULT_RTOL: Q32 = Q32::from_bits(42_950);
/// `np.isclose`'s default `atol = 1e-8`, quantized: `round(1e-8 * 2^32)`.
pub const DEFAULT_ATOL: Q32 = Q32::from_bits(43);

/// `np.isclose(a, b, rtol, atol)`: `|a - b| <= atol + rtol * |b|` (not
/// symmetric in `a` and `b`). Q32 values are never nan, so `equal_nan` has no
/// effect. Panics where Zinnia has no proof: `|a - b|` or `|b|` of `2^31` or
/// more, which has no Q32 code (including `|i64::MIN|` of the codes).
pub fn isclose(a: Q32, b: Q32, rtol: Q32, atol: Q32) -> bool {
    let gap = a.checked_sub(b).and_then(Q32::checked_abs).expect("isclose: |a - b| has no Q32 value");
    let magnitude = b.checked_abs().expect("isclose: |b| has no Q32 value");
    gap <= atol + rtol * magnitude
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// `np.argsort(values, kind='stable')`: the indices that sort `values`, ties
/// in index order.
pub fn argsort<T: Ord + Copy>(values: &[T]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by_key(|&i| values[i]);
    order
}

/// `np.argsort(a, axis=0, kind='stable')` of an `R x C` matrix, by column:
/// `result[j]` holds the row indices that sort column `j`, ties in row order,
/// so `result[j][i] == np.argsort(a, axis=0)[i, j]`.
pub fn argsort_axis0<T: Ord + Copy, const R: usize, const C: usize>(
    a: &[[T; C]; R],
) -> [[usize; R]; C] {
    core::array::from_fn(|j| {
        let mut order: [usize; R] = core::array::from_fn(|i| i);
        order.sort_by_key(|&i| a[i][j]);
        order
    })
}

// ---------------------------------------------------------------------------
// Matrix products (ndarray). Each output element is one fused dot product:
// raw products accumulated in i128 and rescaled once, as Zinnia's
// `product_sum` and NumPy's single rounding of a dot product.
// ---------------------------------------------------------------------------

/// `a @ b` for 1-D `a` and `b` (NumPy `dot`/`inner` of vectors).
pub fn inner(a: ArrayView1<Q32>, b: ArrayView1<Q32>) -> Q32 {
    assert_eq!(a.len(), b.len(), "shapes not aligned");
    dot(a.iter().copied().zip(b.iter().copied()))
}

/// `a @ b` for 2-D `a` and `b`.
pub fn matmul(a: ArrayView2<Q32>, b: ArrayView2<Q32>) -> Array2<Q32> {
    assert_eq!(a.ncols(), b.nrows(), "shapes not aligned");
    Array2::from_shape_fn((a.nrows(), b.ncols()), |(i, j)| inner(a.row(i), b.column(j)))
}

/// `a @ x` for 2-D `a` and 1-D `x`.
pub fn matvec(a: ArrayView2<Q32>, x: ArrayView1<Q32>) -> Array1<Q32> {
    a.rows().into_iter().map(|row| inner(row, x)).collect()
}

/// `x @ a` for 1-D `x` and 2-D `a`.
pub fn vecmat(x: ArrayView1<Q32>, a: ArrayView2<Q32>) -> Array1<Q32> {
    a.columns().into_iter().map(|column| inner(x, column)).collect()
}

// ---------------------------------------------------------------------------
// Bounding-box crop
// ---------------------------------------------------------------------------

/// The first to last row and the first to last column of `a` that hold a
/// cell where `hit` is true, or `None` if no cell does: the per-axis min and
/// max of `np.argwhere(mask)`, equally the ranges that
/// `np.flatnonzero(mask.sum(axis=1))` and `np.flatnonzero(mask.sum(axis=0))`
/// span.
pub fn bounding_box<T: Copy, const R: usize, const C: usize>(
    a: &[[T; C]; R],
    hit: impl Fn(T) -> bool,
) -> Option<(RangeInclusive<usize>, RangeInclusive<usize>)> {
    let mut window: Option<(usize, usize, usize, usize)> = None; // top, bottom, left, right
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if hit(v) {
                window = Some(match window {
                    None => (i, i, j, j),
                    Some((top, _, left, right)) => (top, i, left.min(j), right.max(j)),
                });
            }
        }
    }
    window.map(|(top, bottom, left, right)| (top..=bottom, left..=right))
}

/// `a[rows, cols]` for inclusive ranges: the block flattened row-major, with
/// its data-dependent shape `(rows, cols)`.
pub fn crop<T: Copy, const R: usize, const C: usize>(
    a: &[[T; C]; R],
    rows: RangeInclusive<usize>,
    cols: RangeInclusive<usize>,
) -> Shaped<T> {
    let shape = vec![extent(&rows), extent(&cols)];
    let values = a[rows].iter().flat_map(|row| row[cols.clone()].iter().copied()).collect();
    Shaped { shape, values }
}

fn extent(range: &RangeInclusive<usize>) -> i64 {
    (range.end() - range.start() + 1) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Q32 {
        Q32::from_int(n)
    }

    #[test]
    fn indexes_like_numpy() {
        assert_eq!(np_index(-1, 5), 4);
        assert_eq!(np_index(2, 5), 2);
        assert_eq!(insert_position(5, 5), 5);
        assert_eq!(insert_position(-5, 5), 0);
        assert_eq!(insert_position(-1, 5), 4);
        assert_eq!(inverse_permutation(&[2, 0, -2]), [1, 2, 0]);
    }

    #[test]
    #[should_panic(expected = "index out of range")]
    fn index_past_the_end_panics() {
        np_index(5, 5);
    }

    #[test]
    #[should_panic(expected = "insert position out of range")]
    fn insert_past_the_end_panics() {
        insert_position(6, 5);
    }

    #[test]
    #[should_panic(expected = "uninitialized")]
    fn inverse_of_a_non_permutation_panics() {
        inverse_permutation(&[0, 0, 1]);
    }

    #[test]
    fn arg_extrema_take_the_first_occurrence() {
        let values = [q(1), q(3), q(-2), q(3), q(-2)];
        assert_eq!(argmax(&values), 1);
        assert_eq!(argmin(&values), 2);
        assert_eq!(argmax(&[false, true, true]), 1);
    }

    #[test]
    #[should_panic(expected = "empty sequence")]
    fn argmax_of_nothing_panics() {
        argmax::<Q32>(&[]);
    }

    #[test]
    fn mean_and_std() {
        let values = [q(2), q(4), q(4), q(4), q(5), q(5), q(7), q(9)];
        let m = mean(&values);
        assert_eq!(m, q(5));
        assert_eq!(std(&values, m), q(2));
        // The division rounds toward negative infinity: -1/3 -> -0x55555556.
        assert_eq!(mean(&[q(-1), q(0), q(0)]), Q32::from_bits(-1_431_655_766));
    }

    #[test]
    fn isclose_is_relative_to_the_second_argument() {
        let b = q(100_000);
        assert!(isclose(b + q(1), b, DEFAULT_RTOL, DEFAULT_ATOL));
        assert!(!isclose(b + q(2), b, DEFAULT_RTOL, DEFAULT_ATOL));
        assert!(isclose(Q32::from_bits(43), Q32::ZERO, DEFAULT_RTOL, DEFAULT_ATOL));
        assert!(!isclose(Q32::from_bits(43), Q32::ZERO, DEFAULT_RTOL, Q32::ZERO));
    }

    #[test]
    fn isclose_has_no_value_where_a_magnitude_leaves_q32() {
        let half = Q32::from_bits(1 << 31);
        let min = Q32::from_bits(i64::MIN);
        // |a - b| = 2^63 - 1 (codes) is fine, 2^63 is not; nor is |b| = 2^63.
        assert!(!isclose(Q32::from_bits(i64::MIN + (1 << 31) + 1), half, DEFAULT_RTOL, DEFAULT_ATOL));
        let panics = |a: Q32, b: Q32| std::panic::catch_unwind(|| isclose(a, b, DEFAULT_RTOL, DEFAULT_ATOL)).is_err();
        assert!(panics(Q32::from_bits(i64::MIN + (1 << 31)), half));
        assert!(panics(-Q32::ONE, min));
        assert!(panics(Q32::from_bits(i64::MAX), -Q32::ONE));
    }

    #[test]
    fn argsort_is_stable() {
        assert_eq!(argsort(&[3, -1, 2, -1, 0]), [1, 3, 4, 2, 0]);
        let a = [[q(2), q(0)], [q(1), q(0)], [q(3), q(-1)]];
        assert_eq!(argsort_axis0(&a), [[1, 0, 2], [2, 0, 1]]);
    }

    #[test]
    fn crops_to_the_bounding_box() {
        let a = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0]];
        let (rows, cols) = bounding_box(&a, |v| v != 0).unwrap();
        assert_eq!((rows.clone(), cols.clone()), (1..=2, 1..=3));
        assert_eq!(
            crop(&a, rows, cols),
            Shaped { shape: vec![2, 3], values: vec![1, 0, 0, 0, 0, 2] }
        );
        assert_eq!(bounding_box(&a, |v| v > 5), None);
    }
}
