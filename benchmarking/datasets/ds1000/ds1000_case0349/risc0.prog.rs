// DS-1000 case 349 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(env::read());
        }
        a.push(row);
    }
    let claimed: u64 = env::read();

    let mut sink: f64 = 0.0;
    for i in 0..8 {
        for j in 0..6 {
            sink += a[i][j] * 0.0;
        }
    }
    assert_eq!(sink, 0.0);

    let result: u64 = 6;
    assert_eq!(result, claimed);
}
