// DS-1000 case 361 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [[f64; 6]; 6] = [[0.0; 6]; 6];
    for i in 0..6 { for j in 0..6 { a[i][j] = env::read(); } }
    let mut claimed: [[f64; 4]; 6] = [[0.0; 4]; 6];
    for i in 0..6 { for j in 0..4 { claimed[i][j] = env::read(); } }

    let kept: [usize; 4] = [1, 3, 4, 5];
    let mut result: [[f64; 4]; 6] = [[0.0; 4]; 6];
    for i in 0..6 {
        for (jo, &j) in kept.iter().enumerate() {
            result[i][jo] = a[i][j];
        }
    }
    for i in 0..6 { for j in 0..4 {
        assert!((result[i][j] - claimed[i][j]).abs() < 1e-3);
    } }
}
