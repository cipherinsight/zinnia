// DS-1000 case 303 — SP1 guest program.
// Kernel: B = A[:col].reshape(-1, ncol) where col = (23 // ncol) * ncol on Float[23].
// Per Zinnia IR: per-cell BoolCast/Select (keep cells with i < col_keep else 0) and
// expose shape metadata (rows, ncol, total, rank).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [f64; 23] = [0.0; 23];
    for i in 0..23 {
        a[i] = sp1_zkvm::io::read::<f64>();
    }
    let ncol: i64 = sp1_zkvm::io::read::<i64>();

    assert!(ncol != 0);
    let rows: i64 = 23 / ncol;
    let col_keep: i64 = rows * ncol;

    // Per-cell select: keep a[i] if i < col_keep else 0.
    let mut out: [f64; 23] = [0.0; 23];
    for i in 0..23 {
        let in_range = (i as i64) < col_keep;
        out[i] = if in_range { a[i] } else { 0.0 };
    }

    // Read claimed outputs and assert equality.
    for i in 0..23 {
        let claim_v: f64 = sp1_zkvm::io::read::<f64>();
        assert!((out[i] - claim_v).abs() < 1e-12);
    }
    // Shape metadata: (rows, ncol, total=col_keep, rank=2).
    let claim_rows: i64 = sp1_zkvm::io::read::<i64>();
    let claim_ncol: i64 = sp1_zkvm::io::read::<i64>();
    let claim_total: i64 = sp1_zkvm::io::read::<i64>();
    let claim_rank: i64 = sp1_zkvm::io::read::<i64>();
    assert_eq!(rows, claim_rows);
    assert_eq!(ncol, claim_ncol);
    assert_eq!(col_keep, claim_total);
    assert_eq!(2, claim_rank);
}
