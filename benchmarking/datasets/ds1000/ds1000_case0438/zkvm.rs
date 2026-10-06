//! DS-1000 case 0438: Pearson correlation of two integer series,
//! `np.corrcoef(post, distance)[0][1]`.
//!
//! ```python
//! def kernel(post: Integer[Array, "4"], distance: Integer[Array, "4"]):
//!     result = np.corrcoef(post, distance)[0][1]
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 4;

/// `B * C` must lie below `2^62` for `sqrt(B * C)` to have a Q32 value.
const PRODUCT_LIMIT: i128 = 1 << 62;

pub type Output = Q32;

pub fn main(input: &mut impl Input) -> Output {
    let post: [i64; N] = input.read_array();
    let distance: [i64; N] = input.read_array();
    pearson(&post, &distance)
}

/// `r = A / sqrt(B * C)` from the exact integer sums `A = n Sxy - Sx Sy`,
/// `B = n Sxx - Sx^2`, `C = n Syy - Sy^2` (NumPy's centring and its
/// normalisation factors cancel), rounded as Zinnia and the circuit baselines
/// do: `floor(A * 2^64 / floor(sqrt(B * C * 2^64)))`. `|r| <= 1` holds exactly
/// (Cauchy-Schwarz and a floor root), so NumPy's clip never acts.
///
/// The sums come from the scaled deviations `d_i = n x_i - Sx`, which are
/// exact in `i128` (`|d_i| < 2^66`): `sum(dx dy) = n A`, `sum(dx^2) = n B`,
/// `sum(dy^2) = n C`. No result (a panic) where Zinnia has no proof:
/// `B * C = 0` (a constant series; NumPy gives nan) or `B * C >= 2^62`
/// (`sqrt(B * C)` leaves the Q32 range). A checked operation that overflows
/// already means `n B` or `B * C` is at least `2^127`, outside that domain.
fn pearson(x: &[i64; N], y: &[i64; N]) -> Q32 {
    let (dx, dy) = (deviations(x), deviations(y));
    let b = sum_of_squares(&dx) / N as i128;
    let c = sum_of_squares(&dy) / N as i128;
    let bc = b.checked_mul(c).filter(|bc| (1..PRODUCT_LIMIT).contains(bc));
    let bc = bc.expect("corrcoef: B * C outside [1, 2^62): a constant series or no Q32 root");
    // Inside the domain |dx_i dy_i| <= n sqrt(B C) < 2^33: no overflow.
    let a = dx.iter().zip(&dy).map(|(p, q)| p * q).sum::<i128>() / N as i128;
    let root = ((bc as u128) << 64).isqrt() as i128;
    Q32::from_bits((a << 64).div_euclid(root) as i64)
}

/// `n x_i - sum(x)` for every `i`.
fn deviations(x: &[i64; N]) -> [i128; N] {
    let sum: i128 = x.iter().map(|&v| v as i128).sum();
    x.map(|v| N as i128 * v as i128 - sum)
}

/// `sum(d_i^2)`, panicking on overflow (see [`pearson`]).
fn sum_of_squares(d: &[i128; N]) -> i128 {
    d.iter()
        .try_fold(0i128, |acc, &v| v.checked_mul(v).and_then(|sq| acc.checked_add(sq)))
        .expect("corrcoef: a variance overflows i128, so B * C >= 2^62")
}
