// DS-1000 case 301 — SP1 guest program.
// Kernel: np.reshape(A, (-1, ncol)) on Float[20] with runtime ncol.
// Layout-only; assert divisibility and ncol != 0 then expose 20 cells + shape.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [f64; 20] = [0.0; 20];
    for i in 0..20 {
        a[i] = sp1_zkvm::io::read::<f64>();
    }
    let ncol: i64 = sp1_zkvm::io::read::<i64>();
    let claim_rows: i64 = sp1_zkvm::io::read::<i64>();
    let claim_ncol: i64 = sp1_zkvm::io::read::<i64>();

    assert!(ncol != 0);
    let rows: i64 = 20 / ncol;
    let rem: i64 = 20 % ncol;
    assert_eq!(rem, 0);

    assert_eq!(rows, claim_rows);
    assert_eq!(ncol, claim_ncol);

    // Cells are layout-only — verify each cell against the claim (passed via stdin).
    for i in 0..20 {
        let claim_v: f64 = sp1_zkvm::io::read::<f64>();
        assert!((a[i] - claim_v).abs() < 1e-12);
    }
    // Expose the static shape metadata too.
    let claim_total: i64 = sp1_zkvm::io::read::<i64>();
    let claim_rank: i64 = sp1_zkvm::io::read::<i64>();
    assert_eq!(claim_total, 20);
    assert_eq!(claim_rank, 2);
}
