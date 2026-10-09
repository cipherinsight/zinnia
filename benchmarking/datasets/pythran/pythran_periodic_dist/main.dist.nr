// Source directed pairwise differences and two sequential periodic masks.
// Every directed difference is squared; no pair-symmetry algebraic shortcut.
// @outputs: float[N * N]; float[N * N]; float[N * N]; float[N * N]
use noir_hw::{check_inputs, check_input, canonical_lt, canonical_gt, mul, sqrt, SCALE};
// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0
fn axis(c: [Field; N], l: Field, periodic: bool) -> [Field; N * N] {
    let lf = l * SCALE;
    let half = l * (SCALE / 2);
    check_input(half);
    check_input(-half);
    let mut entries: [Field; N * N] = [0; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut delta = c[j] - c[i];
            if periodic & canonical_gt(delta, half) { delta -= lf; }
            if periodic & canonical_lt(delta, -half) { delta += lf; }
            entries[i * N + j] = delta;
        }
    }
    entries
}
fn main(x: [Field; N], y: [Field; N], z: [Field; N], L: Field,
        periodicX: bool, periodicY: bool, periodicZ: bool)
        -> pub ([Field; N * N], [Field; N * N], [Field; N * N], [Field; N * N]) {
    check_inputs(x); check_inputs(y); check_inputs(z); check_input(L);
    let dx = axis(x, L, periodicX);
    let dy = axis(y, L, periodicY);
    let dz = axis(z, L, periodicZ);
    let mut d: [Field; N * N] = [0; N * N];
    for at in 0..N * N {
        let square_x = mul(dx[at], dx[at]);
        let square_y = mul(dy[at], dy[at]);
        let square_z = mul(dz[at], dz[at]);
        let sum = square_x + square_y + square_z;
        // Noir source sum >= 0 is an asserted 64-bit nonnegative gap.
        sum.assert_max_bit_size::<64>();
        let root = sqrt(sum);
        d[at] = if root == 0 { -SCALE } else { root };
    }
    (d, dx, dy, dz)
}
