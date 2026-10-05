// DS-1000 case 389 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 6;
    const NROWS: usize = 3;
    const NCOLS: usize = 4;

    let mut a: Vec<Vec<Vec<f64>>> = Vec::with_capacity(N);
    for _ in 0..N {
        let mut plane: Vec<Vec<f64>> = Vec::with_capacity(NROWS);
        for _ in 0..NROWS {
            let mut row: Vec<f64> = Vec::with_capacity(NCOLS);
            for _ in 0..NCOLS { row.push(env::read()); }
            plane.push(row);
        }
        a.push(plane);
    }
    let h: u64 = env::read();
    let w: u64 = env::read();
    let hu: usize = h as usize;
    let wu: usize = w as usize;

    let mut claimed: Vec<f64> = Vec::with_capacity(hu * wu);
    for _ in 0..(hu * wu) { claimed.push(env::read()); }

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
