// DS-1000 case 358 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a = [[[0.0f64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { a[i][j][k] = env::read(); } } }
    let mut b = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { b[i][j][k] = env::read(); } } }
    let mut claimed = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { claimed[i][j][k] = env::read(); } } }

    let mut sums = [0.0f64; 3];
    for i in 0..3 {
        let mut s = 0.0f64;
        for j in 0..3 { for k in 0..3 { s += a[i][j][k]; } }
        sums[i] = s;
    }
    let mut idx = [0usize, 1, 2];
    if sums[idx[0]] > sums[idx[1]] { idx.swap(0, 1); }
    if sums[idx[1]] > sums[idx[2]] { idx.swap(1, 2); }
    if sums[idx[0]] > sums[idx[1]] { idx.swap(0, 1); }

    let mut result = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        result[i][j][k] = b[idx[i]][j][k];
    } } }
    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        assert_eq!(result[i][j][k], claimed[i][j][k]);
    } } }
}
