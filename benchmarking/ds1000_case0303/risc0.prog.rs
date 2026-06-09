// DS-1000 case 303 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [f64; 23] = [0.0; 23];
    for i in 0..23 {
        a[i] = env::read();
    }
    let ncol: i64 = env::read();

    assert!(ncol != 0);
    let rows: i64 = 23 / ncol;
    let col_keep: i64 = rows * ncol;

    let mut out: [f64; 23] = [0.0; 23];
    for i in 0..23 {
        let in_range = (i as i64) < col_keep;
        out[i] = if in_range { a[i] } else { 0.0 };
    }

    for i in 0..23 {
        let claim_v: f64 = env::read();
        assert!((out[i] - claim_v).abs() < 1e-12);
    }
    let claim_rows: i64 = env::read();
    let claim_ncol: i64 = env::read();
    let claim_total: i64 = env::read();
    let claim_rank: i64 = env::read();
    assert_eq!(rows, claim_rows);
    assert_eq!(ncol, claim_ncol);
    assert_eq!(col_keep, claim_total);
    assert_eq!(2, claim_rank);
}
