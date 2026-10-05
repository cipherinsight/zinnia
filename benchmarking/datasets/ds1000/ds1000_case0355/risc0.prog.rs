// DS-1000 case 355 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a = [[[0.0f64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { a[i][j][k] = env::read(); } } }
    let mut b = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { b[i][j][k] = env::read(); } } }
    let mut claimed = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { claimed[i][j][k] = env::read(); } } }

    let mut sort_indices = [[[0usize; 3]; 3]; 3];
    for j in 0..3 {
        for k in 0..3 {
            let mut idx = [0usize, 1, 2];
            if a[idx[0]][j][k] > a[idx[1]][j][k] { idx.swap(0, 1); }
            if a[idx[1]][j][k] > a[idx[2]][j][k] { idx.swap(1, 2); }
            if a[idx[0]][j][k] > a[idx[1]][j][k] { idx.swap(0, 1); }
            for i in 0..3 { sort_indices[i][j][k] = idx[i]; }
        }
    }

    let mut c = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        c[i][j][k] = b[sort_indices[i][j][k]][j][k];
    } } }

    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        assert_eq!(c[i][j][k], claimed[i][j][k]);
    } } }
}
