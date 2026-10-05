// DS-1000 case 359 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [[f64; 3]; 3] = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { a[i][j] = env::read(); } }
    let mut claimed: [[f64; 2]; 3] = [[0.0; 2]; 3];
    for i in 0..3 { for j in 0..2 { claimed[i][j] = env::read(); } }

    let kept: [usize; 2] = [0, 1];
    let mut result: [[f64; 2]; 3] = [[0.0; 2]; 3];
    for i in 0..3 {
        for (jo, &j) in kept.iter().enumerate() {
            result[i][jo] = a[i][j];
        }
    }

    for i in 0..3 { for j in 0..2 {
        assert!((result[i][j] - claimed[i][j]).abs() < 1e-3);
    } }
}
