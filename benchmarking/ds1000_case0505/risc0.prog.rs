// DS-1000 case 505 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<Vec<Vec<f64>>> = Vec::with_capacity(13);
    for _ in 0..13 {
        let mut plane: Vec<Vec<f64>> = Vec::with_capacity(13);
        for _ in 0..13 {
            let mut row: Vec<f64> = Vec::with_capacity(5);
            for _ in 0..5 {
                row.push(env::read());
            }
            plane.push(row);
        }
        a.push(plane);
    }
    let mut b: Vec<Vec<u64>> = Vec::with_capacity(13);
    for _ in 0..13 {
        let mut row: Vec<u64> = Vec::with_capacity(13);
        for _ in 0..13 {
            row.push(env::read());
        }
        b.push(row);
    }
    let claimed: f64 = env::read();

    let mut arr: Vec<Vec<f64>> = Vec::with_capacity(13);
    for i in 0..13 {
        let mut row: Vec<f64> = Vec::with_capacity(13);
        for j in 0..13 {
            let k = b[i][j] as usize;
            row.push(a[i][j][k]);
        }
        arr.push(row);
    }

    let mut sum_a: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            for k in 0..5 {
                sum_a += a[i][j][k];
            }
        }
    }
    let mut sum_arr: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            sum_arr += arr[i][j];
        }
    }
    let result: f64 = sum_a - sum_arr;

    assert!((result - claimed).abs() < 1e-6);
}
