// DS-1000 case 389 — SP1 guest program.
// Kernel: a.reshape(h/nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)
// on Float[6, 3, 4]; sample h=6, w=12. The reshape splits the leading 6 into (2, 3),
// swap interleaves the second and third axes, then flattens back to 2D.
//
// Index mapping (verified): out[out_i][out_j] = a[i0*3 + i2][i1][i3] where
//   out_idx = out_i*w + out_j, and (i0, i1, i2, i3) decode (2, 3, 3, 4) row-major.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 6;
    const NROWS: usize = 3;
    const NCOLS: usize = 4;

    let mut a: Vec<Vec<Vec<f64>>> = Vec::with_capacity(N);
    for _ in 0..N {
        let mut plane: Vec<Vec<f64>> = Vec::with_capacity(NROWS);
        for _ in 0..NROWS {
            let mut row: Vec<f64> = Vec::with_capacity(NCOLS);
            for _ in 0..NCOLS { row.push(sp1_zkvm::io::read::<f64>()); }
            plane.push(row);
        }
        a.push(plane);
    }
    let h: u64 = sp1_zkvm::io::read::<u64>();
    let w: u64 = sp1_zkvm::io::read::<u64>();
    let hu: usize = h as usize;
    let wu: usize = w as usize;

    let mut claimed: Vec<f64> = Vec::with_capacity(hu * wu);
    for _ in 0..(hu * wu) { claimed.push(sp1_zkvm::io::read::<f64>()); }

    // Recompute using the same index permutation.
    let mut result: Vec<f64> = Vec::with_capacity(hu * wu);
    for out_i in 0..hu {
        for out_j in 0..wu {
            let out_idx = out_i * wu + out_j;
            let i0 = out_idx / 36;
            let i1 = (out_idx / 12) % 3;
            let i2 = (out_idx / 4) % 3;
            let i3 = out_idx % 4;
            result.push(a[i0 * NROWS + i2][i1][i3]);
        }
    }

    for i in 0..(hu * wu) {
        assert!((result[i] - claimed[i]).abs() < 1e-9);
    }
}
