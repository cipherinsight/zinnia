// DS-1000 case 353 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [[i64; 3]; 2] = [[0; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            a[i][j] = env::read();
        }
    }
    let mut b: [[i64; 3]; 7] = [[0; 3]; 7];
    for i in 0..7 {
        for j in 0..3 {
            b[i][j] = env::read();
        }
    }

    let n_claimed: u64 = env::read();
    let mut claimed: Vec<[i64; 3]> = Vec::with_capacity(n_claimed as usize);
    for _ in 0..n_claimed {
        let mut row = [0i64; 3];
        for j in 0..3 { row[j] = env::read(); }
        claimed.push(row);
    }

    let mut dims = [0i64; 3];
    for k in 0..3 {
        let mut m = a[0][k];
        for i in 1..2 { if a[i][k] > m { m = a[i][k]; } }
        for i in 0..7 { if b[i][k] > m { m = b[i][k]; } }
        dims[k] = m + 1;
    }
    let mut a_idx = [0i64; 2];
    for r in 0..2 {
        a_idx[r] = ((a[r][0] * dims[1]) + a[r][1]) * dims[2] + a[r][2];
    }
    let mut b_idx = [0i64; 7];
    for r in 0..7 {
        b_idx[r] = ((b[r][0] * dims[1]) + b[r][1]) * dims[2] + b[r][2];
    }
    let mut in_b = [false; 2];
    for r in 0..2 {
        for s in 0..7 {
            if a_idx[r] == b_idx[s] { in_b[r] = true; }
        }
    }
    let mut out: Vec<[i64; 3]> = Vec::new();
    for r in 0..2 {
        if !in_b[r] { out.push([a[r][0], a[r][1], a[r][2]]); }
    }
    assert_eq!(out.len() as u64, n_claimed);
    for r in 0..out.len() {
        for j in 0..3 { assert_eq!(out[r][j], claimed[r][j]); }
    }
}
