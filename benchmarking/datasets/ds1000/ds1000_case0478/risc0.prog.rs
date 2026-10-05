// DS-1000 case 478 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [[i64; 4]; 4] = [[0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = env::read();
        }
    }
    let mut claimed: [i64; 4] = [0; 4];
    for k in 0..4 {
        claimed[k] = env::read();
    }

    let mut result: [i64; 4] = [0; 4];
    for i in 0..4 {
        result[i] = a[i][i];
    }
    let save: [i64; 4] = result;
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = 0;
        }
    }
    for i in 0..4 {
        result[i] = save[i];
    }

    for k in 0..4 {
        assert_eq!(result[k], claimed[k]);
    }
}
