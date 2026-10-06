//! Pythran stone: the Whetstone benchmark (modules 1-11 run `LOOP` times
//! their weight), returning its nominal rating `KIP = 100.0 * LOOP * II`.
//! The modules' results never reach the return value; they are the
//! benchmark's work.
//!
//! ```python
//! def whetstone(loopstart: Integer):
//!     T = 0.499975
//!     T1 = 0.50025
//!     T2 = 2.0
//!     LOOP = loopstart
//!     II = 1
//!     JJ = 1
//!     while JJ <= II:
//!         N1 = 0; N2 = 12 * LOOP; N3 = 14 * LOOP; N4 = 345 * LOOP; N6 = 210 * LOOP
//!         N7 = 32 * LOOP; N8 = 899 * LOOP; N9 = 616 * LOOP; N10 = 0; N11 = 93 * LOOP
//!         X1 = 1.0; X2 = -1.0; X3 = -1.0; X4 = -1.0
//!         for I in range(1, N1 + 1):
//!             X1 = (X1 + X2 + X3 - X4) * T
//!             X2 = (X1 + X2 - X3 + X4) * T
//!             X3 = (X1 - X2 + X3 + X4) * T
//!             X4 = (-X1 + X2 + X3 + X4) * T
//!         E1 = [1.0, -1.0, -1.0, -1.0]
//!         for I in range(1, N2 + 1):
//!             (the same four updates on E1[0..4])
//!         for I in range(1, N3 + 1):
//!             PA(E1, T, T2)
//!         J = 1
//!         for I in range(1, N4 + 1):
//!             if J == 1: J = 2
//!             else: J = 3
//!             if J > 2: J = 0
//!             else: J = 1
//!             if J < 1: J = 1
//!             else: J = 0
//!         J = 1; K = 2; L = 3
//!         for I in range(1, N6 + 1):
//!             J = J * (K - J) * (L - K)
//!             K = L * K - (L - J) * K
//!             L = (L - K) * (K + J)
//!             E1[L - 2] = J + K + L
//!             E1[K - 2] = J * K * L
//!         X = 0.5; Y = 0.5
//!         for I in range(1, N7 + 1):
//!             X = T * DATAN(T2 * DSIN(X) * DCOS(X) / (DCOS(X + Y) + DCOS(X - Y) - 1.0))
//!             Y = T * DATAN(T2 * DSIN(Y) * DCOS(Y) / (DCOS(X + Y) + DCOS(X - Y) - 1.0))
//!         X = 1.0; Y = 1.0; Z = 1.0
//!         for I in range(1, N8 + 1):
//!             Z = P3(X, Y, T, T2)
//!         J = 1; K = 2; L = 3
//!         E1[0] = 1.0; E1[1] = 2.0; E1[2] = 3.0
//!         for I in range(1, N9 + 1):
//!             P0(E1, J, K, L)
//!         J = 2; K = 3
//!         for I in range(1, N10 + 1):
//!             J = J + K; K = J + K; J = K - J; K = K - J - J
//!         X = 0.75
//!         for I in range(1, N11 + 1):
//!             X = DSQRT(DEXP(DLOG(X) / T1))
//!         JJ += 1
//!     KIP = 100.0 * LOOP * II
//!     return KIP
//!
//! def PA(E, T, T2):           # six rounds of the E1 updates, the last one / T2
//! def P0(E1, J, K, L):        # E1[J-1] = E1[K-1]; E1[K-1] = E1[L-1]; E1[L-1] = E1[J-1]
//! def P3(X, Y, T, T2):        # X1 = T * (X + Y); Y1 = T * (X1 + Y); return (X1 + Y1) / T2
//! ```
//!
//! `DSIN`, `DCOS`, `DATAN`, `DLOG`, `DEXP`, `DSQRT` are Python's `math`
//! functions.
use crate::prelude::*;

/// `round(0.499975 * 2^32)`.
const T: Q32 = Q32::from_bits(2_147_376_274);
/// `round(0.50025 * 2^32)`.
const T1: Q32 = Q32::from_bits(2_148_557_390);
const T2: Q32 = Q32::from_int(2);

/// The four mixing updates of modules 1 and 2, in place.
fn mix(e: &mut [Q32; 4]) {
    e[0] = (e[0] + e[1] + e[2] - e[3]) * T;
    e[1] = (e[0] + e[1] - e[2] + e[3]) * T;
    e[2] = (e[0] - e[1] + e[2] + e[3]) * T;
    e[3] = (-e[0] + e[1] + e[2] + e[3]) * T;
}

/// Module 3's procedure `PA`.
fn pa(e: &mut [Q32; 4]) {
    for _ in 0..6 {
        e[0] = (e[0] + e[1] + e[2] - e[3]) * T;
        e[1] = (e[0] + e[1] - e[2] + e[3]) * T;
        e[2] = (e[0] - e[1] + e[2] + e[3]) * T;
        e[3] = (-e[0] + e[1] + e[2] + e[3]) / T2;
    }
}

/// Module 7's update of `V` (`X`, then `Y` with the new `X`):
/// `T * DATAN(T2 * DSIN(V) * DCOS(V) / (DCOS(X + Y) + DCOS(X - Y) - 1.0))`.
fn trig_step(v: Q32, x: Q32, y: Q32) -> Q32 {
    T * (T2 * v.sin() * v.cos() / ((x + y).cos() + (x - y).cos() - Q32::ONE)).atan()
}

/// Module 8's procedure `P3`.
fn p3(x: Q32, y: Q32) -> Q32 {
    let x1 = T * (x + y);
    let y1 = T * (x1 + y);
    (x1 + y1) / T2
}

/// Module 9's procedure `P0` (Python list indices: negative ones wrap).
fn p0(e: &mut [Q32; 4], j: i64, k: i64, l: i64) {
    let at = |i: i64| np_index(i - 1, 4);
    e[at(j)] = e[at(k)];
    e[at(k)] = e[at(l)];
    e[at(l)] = e[at(j)];
}

pub mod whetstone {
    use super::*;

    pub type Output = Q32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let loop_count: i64 = args.scalar();
        let ii = 1;
        let mut jj = 1;
        while jj <= ii {
            let n1 = 0;
            let n2 = 12 * loop_count;
            let n3 = 14 * loop_count;
            let n4 = 345 * loop_count;
            let n6 = 210 * loop_count;
            let n7 = 32 * loop_count;
            let n8 = 899 * loop_count;
            let n9 = 616 * loop_count;
            let n10 = 0;
            let n11 = 93 * loop_count;

            // Module 1: simple identifiers (no iterations: N1 = 0).
            let mut x = [Q32::ONE, -Q32::ONE, -Q32::ONE, -Q32::ONE];
            for _ in 1..=n1 {
                mix(&mut x);
            }

            // Module 2: array elements.
            let mut e1 = [Q32::ONE, -Q32::ONE, -Q32::ONE, -Q32::ONE];
            for _ in 1..=n2 {
                mix(&mut e1);
            }

            // Module 3: array as parameter.
            for _ in 1..=n3 {
                pa(&mut e1);
            }

            // Module 4: conditional jumps.
            let mut j = 1;
            for _ in 1..=n4 {
                j = if j == 1 { 2 } else { 3 };
                j = if j > 2 { 0 } else { 1 };
                j = if j < 1 { 1 } else { 0 };
            }

            // Module 6: integer arithmetic (the list stores the integers).
            let (mut j, mut k, mut l) = (1i64, 2i64, 3i64);
            for _ in 1..=n6 {
                j = j * (k - j) * (l - k);
                k = l * k - (l - j) * k;
                l = (l - k) * (k + j);
                e1[np_index(l - 2, 4)] = Q32::from_int(j + k + l);
                e1[np_index(k - 2, 4)] = Q32::from_int(j * k * l);
            }

            // Module 7: trigonometric functions.
            let half = Q32::ONE / 2;
            let (mut x, mut y) = (half, half);
            for _ in 1..=n7 {
                x = trig_step(x, x, y);
                y = trig_step(y, x, y);
            }

            // Module 8: procedure calls.
            let (x, y) = (Q32::ONE, Q32::ONE);
            let mut _z = Q32::ONE;
            for _ in 1..=n8 {
                _z = p3(x, y);
            }

            // Module 9: array references.
            let (j, k, l) = (1, 2, 3);
            e1[0] = Q32::ONE;
            e1[1] = Q32::from_int(2);
            e1[2] = Q32::from_int(3);
            for _ in 1..=n9 {
                p0(&mut e1, j, k, l);
            }

            // Module 10: integer arithmetic (no iterations: N10 = 0).
            let (mut j, mut k) = (2i64, 3i64);
            for _ in 1..=n10 {
                j += k;
                k += j;
                j = k - j;
                k = k - j - j;
            }

            // Module 11: standard functions.
            let mut x = Q32::from_int(3) / 4;
            for _ in 1..=n11 {
                x = (x.ln() / T1).exp().sqrt();
            }

            jj += 1;
        }
        Q32::from_int(100) * loop_count * ii
    }
}
