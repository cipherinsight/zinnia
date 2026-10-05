// DS-1000 case 302 — SP1 guest program.
// Kernel: np.reshape(A, (nrow, -1)) on Float[20] with runtime nrow.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [f64; 20] = [0.0; 20];
    for i in 0..20 {
        a[i] = sp1_zkvm::io::read::<f64>();
    }
    let nrow: i64 = sp1_zkvm::io::read::<i64>();
    let claim_nrow: i64 = sp1_zkvm::io::read::<i64>();
    let claim_ncol: i64 = sp1_zkvm::io::read::<i64>();

    assert!(nrow != 0);
    let ncol: i64 = 20 / nrow;
    let rem: i64 = 20 % nrow;
    assert_eq!(rem, 0);

    assert_eq!(nrow, claim_nrow);
    assert_eq!(ncol, claim_ncol);

    for i in 0..20 {
        let claim_v: f64 = sp1_zkvm::io::read::<f64>();
        assert!((a[i] - claim_v).abs() < 1e-12);
    }
    let claim_total: i64 = sp1_zkvm::io::read::<i64>();
    let claim_rank: i64 = sp1_zkvm::io::read::<i64>();
    assert_eq!(claim_total, 20);
    assert_eq!(claim_rank, 2);
}
