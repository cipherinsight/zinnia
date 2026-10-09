// pythran_periodic_dist__dist (Pythran periodic_dist), Noir baseline.
// Kernel: for particles at `x, y, z: Float[N]` in a box of integer side L,
// dx[i, j] = x[j] - x[i] (likewise dy, dz); along each axis whose boolean
// flag is set, dx[dx > L / 2] -= L, then dx[dx < -L / 2] += L;
// d = sqrt(dx**2 + dy**2 + dz**2) with d[d == 0] = -1. Returns
// (d, dx, dy, dz).
// Per axis, with h = flag * L / 2 (a Q32 value, window-checked: NumPy
// computes L / 2 when the flag is set), f = flag * L and v = c[j] - c[i] for
// i < j, the two comparisons a = v > h and b = v < -h decide both entries:
// the second update sees v - L only when a held, and then v - L < -h is
// false, so dx[i, j] = v + f (b - a - a b) and, as -v > h is b,
// dx[j, i] = -v + f (a - b - a b) = -dx[i, j] - 2 f a b. Each comparison is a
// hinted bit pinned by one range check: for an integer t in (-2^66, 2^66),
// t > 0 holds iff t - 1 lies in [0, 2^CMP_BITS) and fails iff -t does
// (CMP_BITS = 70, five 14-bit limbs: no range table beyond the 64-bit
// checks'); v - h and -v - h lie there since c and h are in the window, so v
// needs no window check of its own. The diagonal (v = 0) is -f [h < 0] for
// every i, computed once, as is its d. The squares are rounded one by one
// (NumPy materializes dx**2). With F the Q32 code of f (2^32 f),
// dx[j, i]^2 = dx[i, j]^2 + 4 F a b (dx[i, j] + F), and
// 4 F a b (dx[i, j] + F) / 2^32 is an integer, so floor(dx[j, i]^2 / 2^32) is
// floor(dx[i, j]^2 / 2^32) plus that integer: one rescaled product per pair,
// not two. d == 0 is tested on d itself.
// @outputs: float[N * N]; float[N * N]; float[N * N]; float[N * N]

use noir_hw::{check_input, check_inputs, is_neg, mul, sqrt, window, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0

global CMP_BITS: u32 = 70;

unconstrained fn is_positive_hint(t: Field) -> bool {
    // t is a small signed integer: positive iff nonzero and below p / 2.
    (t != 0) & t.lt(2.pow_32(126))
}

// t > 0 for an integer t in (-2^66, 2^66).
fn is_positive(t: Field) -> bool {
    // Safety: pinned by the range check (t - 1 or -t in [0, 2^CMP_BITS)).
    let positive = unsafe { is_positive_hint(t) };
    let checked = if positive { t - 1 } else { -t };
    checked.assert_max_bit_size::<CMP_BITS>();
    positive
}

// One axis after the periodic updates under `flag`: c[j] - c[i] and the
// rounded squares.
fn axis(c: [Field; N], flag: bool, l: Field) -> ([[Field; N]; N], [[Field; N]; N]) {
    let f = (flag as Field) * l;
    let f_float = f * SCALE;
    let h = f_float / 2;
    let _ = window(h);
    // The diagonal: v = 0, a = b = [h < 0].
    let diagonal = -f_float * (is_neg(h) as Field);
    let mut e: [[Field; N]; N] = [[diagonal; N]; N];
    let mut squares: [[Field; N]; N] = [[mul(diagonal, diagonal); N]; N];
    for i in 0..N {
        for j in i + 1..N {
            let v = c[j] - c[i];
            let a = is_positive(v - h) as Field;
            let b = is_positive(-v - h) as Field;
            let ab = a * b;
            let forward = v + f_float * (b - a - ab);
            e[i][j] = forward;
            e[j][i] = -forward - 2 * f_float * ab;
            let square = mul(forward, forward);
            squares[i][j] = square;
            squares[j][i] = square + 4 * f * ab * (forward + f_float);
        }
    }
    (e, squares)
}

// sqrt(sx + sy + sz), or -1 where it is 0.
fn distance(sx: Field, sy: Field, sz: Field) -> Field {
    let root = sqrt(sx + sy + sz);
    if root == 0 { -SCALE } else { root }
}

fn main(
    x: [Field; N],
    y: [Field; N],
    z: [Field; N],
    L: Field,
    periodicX: bool,
    periodicY: bool,
    periodicZ: bool,
) -> pub ([[Field; N]; N], [[Field; N]; N], [[Field; N]; N], [[Field; N]; N]) {
    // Input domain: every input element is a Q32 code or an int64 in the
    // window (the flags are `bool`s).
    check_inputs(x);
    check_inputs(y);
    check_inputs(z);
    check_input(L);
    let (dx, sx) = axis(x, periodicX, L);
    let (dy, sy) = axis(y, periodicY, L);
    let (dz, sz) = axis(z, periodicZ, L);
    let diagonal = distance(sx[0][0], sy[0][0], sz[0][0]);
    let mut d: [[Field; N]; N] = [[diagonal; N]; N];
    for i in 0..N {
        for j in 0..N {
            if i != j {
                d[i][j] = distance(sx[i][j], sy[i][j], sz[i][j]);
            }
        }
    }
    (d, dx, dy, dz)
}
