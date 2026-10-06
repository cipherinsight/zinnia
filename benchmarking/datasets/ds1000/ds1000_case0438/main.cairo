//! DS-1000 case 0438: Pearson correlation of two integer series,
//! `np.corrcoef(post, distance)[0][1]`.
//!
//!     def kernel(post: Integer[Array, "4"], distance: Integer[Array, "4"]):
//!         result = np.corrcoef(post, distance)[0][1]
//!         return result
// output: q32
use core::num::traits::Sqrt;
use crate::q32::{Q32, floor_div};

const N: u32 = 4;
/// `B * C` must lie below `2^62` for `sqrt(B * C)` to have a Q32 value.
const PRODUCT_LIMIT: i128 = 0x4000000000000000;
const TWO_POW_64: u128 = 0x10000000000000000;

/// `r = A / sqrt(B * C)` from the exact integer sums `A = n Sxy - Sx Sy`,
/// `B = n Sxx - Sx^2`, `C = n Syy - Sy^2` (NumPy's centring and its
/// normalisation factors cancel), rounded as Zinnia and the circuit baselines
/// do: `floor(A * 2^64 / floor(sqrt(B * C * 2^64)))`. `|r| <= 1` holds exactly
/// (Cauchy-Schwarz and a floor root), so NumPy's clip never acts.
///
/// The sums come from the scaled deviations `d_i = n x_i - Sx` (exact in
/// `i128`): `sum(dx dy) = n A`, `sum(dx^2) = n B`, `sum(dy^2) = n C`. No result
/// where Zinnia has no proof: `B * C = 0` (a constant series; NumPy gives nan)
/// or `B * C >= 2^62` (`sqrt(B * C)` leaves the Q32 range). An `i128` overflow,
/// which fails the execution, already means `n B` or `B * C` is at least
/// `2^127`, outside that domain.
#[executable]
fn main(post: Array<i64>, distance: Array<i64>) -> Q32 {
    assert!(post.len() == N && distance.len() == N, "inputs have the wrong length");
    let n: i128 = N.into();
    let (dx, dy) = (deviations(post.span()), deviations(distance.span()));
    let bc = (sum_of_products(dx, dx) / n) * (sum_of_products(dy, dy) / n);
    assert!(1 <= bc && bc < PRODUCT_LIMIT, "B * C outside [1, 2^62)");
    let a = sum_of_products(dx, dy) / n;
    let bc: u128 = bc.try_into().unwrap();
    let root: u64 = (bc * TWO_POW_64).sqrt();
    let scaled_a = a * TWO_POW_64.try_into().unwrap();
    Q32 { bits: floor_div(scaled_a, root.into()).try_into().unwrap() }
}

/// `n x_i - sum(x)` for every `i`.
fn deviations(x: Span<i64>) -> Span<i128> {
    let n: i128 = N.into();
    let mut sum: i128 = 0;
    for v in x {
        let wide: i128 = (*v).into();
        sum += wide;
    }
    let mut out = array![];
    for v in x {
        let wide: i128 = (*v).into();
        out.append(n * wide - sum);
    }
    out.span()
}

/// `sum(a_i * b_i)`.
fn sum_of_products(a: Span<i128>, b: Span<i128>) -> i128 {
    let mut sum: i128 = 0;
    for i in 0..a.len() {
        sum += *a.at(i) * *b.at(i);
    }
    sum
}
