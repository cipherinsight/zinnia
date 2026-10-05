// DS-1000 case 364 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a = [[0.0f64; 10]; 100];
    for i in 0..100 { for j in 0..10 { a[i][j] = env::read(); } }
    let pos: i64 = env::read();
    let mut element = [0.0f64; 10];
    for j in 0..10 { element[j] = env::read(); }
    let mut claimed = [[0.0f64; 10]; 101];
    for i in 0..101 { for j in 0..10 { claimed[i][j] = env::read(); } }

    let pos_u: usize = pos as usize;
    let mut result = [[0.0f64; 10]; 101];
    for i in 0..101 {
        if i < pos_u {
            for j in 0..10 { result[i][j] = a[i][j]; }
        } else if i == pos_u {
            for j in 0..10 { result[i][j] = element[j]; }
        } else {
            for j in 0..10 { result[i][j] = a[i - 1][j]; }
        }
    }
    for i in 0..101 { for j in 0..10 {
        assert!((result[i][j] - claimed[i][j]).abs() < 1e-3);
    } }
}
