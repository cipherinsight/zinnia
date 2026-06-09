// DS-1000 case 302 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [f64; 20] = [0.0; 20];
    for i in 0..20 {
        a[i] = env::read();
    }
    let nrow: i64 = env::read();
    let claim_nrow: i64 = env::read();
    let claim_ncol: i64 = env::read();

    assert!(nrow != 0);
    let ncol: i64 = 20 / nrow;
    let rem: i64 = 20 % nrow;
    assert_eq!(rem, 0);

    assert_eq!(nrow, claim_nrow);
    assert_eq!(ncol, claim_ncol);

    for i in 0..20 {
        let claim_v: f64 = env::read();
        assert!((a[i] - claim_v).abs() < 1e-12);
    }
    let claim_total: i64 = env::read();
    let claim_rank: i64 = env::read();
    assert_eq!(claim_total, 20);
    assert_eq!(claim_rank, 2);
}
