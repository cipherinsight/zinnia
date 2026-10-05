//! NPBench contour_integral: quadrature of a resolvent along a contour; at
//! every point `z`, `X = T(z)^-1` (or `T(z)^-1 Y`) with
//! `T(z) = sum_n z^(slab_per_bc / 2 - n) Ham[n]`, accumulated into `P0` and
//! `z X` into `P1`.
//!
//! ```python
//! def contour_integral(NR: Integer, NM: Integer, slab_per_bc: Integer, Ham: Complex[Array, '3 50 50'], int_pts: Complex[Array, '32'], Y: Complex[Array, '50 150']):
//!     P0 = np.zeros((NR, NM), dtype=np.complex128)
//!     P1 = np.zeros((NR, NM), dtype=np.complex128)
//!     for z in int_pts:
//!         Tz = np.zeros((NR, NR), dtype=np.complex128)
//!         for n in range(slab_per_bc + 1):
//!             zz = np.power(z, slab_per_bc / 2 - n)
//!             Tz += zz * Ham[n]
//!         if NR == NM:
//!             X = np.linalg.inv(Tz)
//!         else:
//!             X = np.linalg.solve(Tz, Y)
//!         if abs(z) < 1.0:
//!             X = -X
//!         P0 += X
//!         P1 += z * X
//!     return (P0, P1)
//! ```
use crate::prelude::*;

/// `np.power(z, p)` for a complex `z` and a real exponent `p`, as NumPy's
/// `npy_cpow` computes it: 1 for `p == 0`; 0 for `z == 0` and `p > 0`; for
/// an integer `p` with `|p| < 100`, binary powering by complex products,
/// then `1 / z^|p|` if `p < 0`; otherwise `exp(p * log z)`.
fn power(z: C32, p: Q32) -> C32 {
    if p == Q32::ZERO {
        return C32::ONE;
    }
    if z == C32::ZERO {
        assert!(p > Q32::ZERO, "zero to a negative power is nan");
        return C32::ZERO;
    }
    let n = p.trunc_to_int();
    if Q32::from_int(n) == p && n.abs() < 100 {
        let (mut result, mut base, mut bits) = (C32::ONE, z, n.unsigned_abs());
        loop {
            if bits & 1 == 1 {
                result *= base;
            }
            bits >>= 1;
            if bits == 0 {
                break;
            }
            base = base * base;
        }
        return if n < 0 { C32::ONE / result } else { result };
    }
    let log = C32::new(z.abs().ln(), z.im.atan2(z.re));
    (log * p).exp()
}

/// `np.linalg.solve(a, b)` (LAPACK `zgesv`): LU factorization of `a` with
/// partial pivoting, applied to the columns of `b`, then back substitution.
/// The pivot of a column is its first entry of largest `|re| + |im|` on or
/// below the diagonal (`izamax`). `np.linalg.inv(a)` is `solve(a, I)`.
fn solve(mut a: Array2<C32>, mut b: Array2<C32>) -> Array2<C32> {
    let n = a.nrows();
    assert!(
        a.ncols() == n && b.nrows() == n,
        "solve: shapes do not match"
    );
    for col in 0..n {
        let sizes: Vec<Q32> = a
            .slice(s![col.., col])
            .iter()
            .map(|v| v.re.abs() + v.im.abs())
            .collect();
        let pivot_row = col + np::argmax(&sizes);
        assert!(a[(pivot_row, col)] != C32::ZERO, "Singular matrix");
        if pivot_row != col {
            for k in 0..n {
                a.swap((col, k), (pivot_row, k));
            }
            for k in 0..b.ncols() {
                b.swap((col, k), (pivot_row, k));
            }
        }
        let pivot = a[(col, col)];
        for row in col + 1..n {
            let factor = a[(row, col)] / pivot;
            for k in col + 1..n {
                let update = factor * a[(col, k)];
                a[(row, k)] -= update;
            }
            for k in 0..b.ncols() {
                let update = factor * b[(col, k)];
                b[(row, k)] -= update;
            }
        }
    }
    for row in (0..n).rev() {
        for k in 0..b.ncols() {
            let mut value = b[(row, k)];
            for m in row + 1..n {
                value -= a[(row, m)] * b[(m, k)];
            }
            b[(row, k)] = value / a[(row, row)];
        }
    }
    b
}

pub mod contour_integral {
    use super::*;

    pub type Output = (Array2<C32>, Array2<C32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let nr: i64 = args.scalar();
        let nm: i64 = args.scalar();
        let slab_per_bc: i64 = args.scalar();
        let ham: Array3<C32> = args.array3();
        let int_pts: Array1<C32> = args.array1();
        let y: Array2<C32> = args.array2();
        let extent = |n: i64| usize::try_from(n).expect("negative dimensions are not allowed");
        let (rows, cols) = (extent(nr), extent(nm));
        let mut p0 = Array2::<C32>::zeros((rows, cols));
        let mut p1 = Array2::<C32>::zeros((rows, cols));
        for &z in &int_pts {
            let mut tz = Array2::<C32>::zeros((rows, rows));
            for n in 0..slab_per_bc + 1 {
                // slab_per_bc / 2 is a float division.
                let zz = power(z, Q32::from_int(slab_per_bc) / 2 - Q32::from_int(n));
                tz.zip_mut_with(&ham.index_axis(Axis(0), n as usize), |t, &h| *t += zz * h);
            }
            let mut x = if nr == nm {
                solve(tz, Array2::eye(rows))
            } else {
                solve(tz, y.clone())
            };
            if z.abs() < Q32::ONE {
                x.mapv_inplace(|v| -v);
            }
            p0 += &x;
            p1.zip_mut_with(&x, |p, &v| *p += z * v);
        }
        (p0, p1)
    }
}
