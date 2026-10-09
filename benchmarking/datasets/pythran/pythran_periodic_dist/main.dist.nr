// Source directed pairwise differences and two sequential periodic masks.
// Every directed difference is squared; no pair-symmetry algebraic shortcut.
// @outputs: float[N * N]; float[N * N]; float[N * N]; float[N * N]
use noir_hw::{check_inputs, check_input, window, mul, sqrt, SCALE};
// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0
fn axis(c: [Field; N], l: Field, periodic: bool) -> [Field; N * N] {
    let lf = if periodic { l * SCALE } else { 0 };
    let half = if periodic { l * (SCALE / 2) } else { 0 };
    let half_window = window(half);
    let negative_half_window = window(-half);
    let mut entries: [Field; N * N] = [0; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut delta = c[j] - c[i];
            if periodic & (window(delta) > half_window) { delta -= lf; }
            if periodic & (window(delta) < negative_half_window) { delta += lf; }
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
        let root = sqrt(square_x + square_y + square_z);
        d[at] = if root == 0 { -SCALE } else { root };
    }
    (d, dx, dy, dz)
}
