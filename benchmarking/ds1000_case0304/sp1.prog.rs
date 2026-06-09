// DS-1000 case 304 — SP1 guest program.
// Kernel: col = (23 // ncol) * ncol; B = A[len(A)-col:][::-1]; B = reshape(B, (-1, ncol)).
// On Float[23]: read 23 cells, compute col_keep, then per-cell select pulling
// in_range[k] = (k < col_keep) and out[k] = A[(len(A)-1) - k] if in_range else 0.

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

    // out[k] = A[len(A)-col + (col-1-k)] = A[len(A) - 1 - k] for k < col_keep, else 0.
    let mut out: [f64; 23] = [0.0; 23];
    for k in 0..23 {
        let in_range = (k as i64) < col_keep;
        let src_idx: usize = 22 - k;
        out[k] = if in_range { a[src_idx] } else { 0.0 };
    }

    for k in 0..23 {
        let claim_v: f64 = sp1_zkvm::io::read::<f64>();
        assert!((out[k] - claim_v).abs() < 1e-12);
    }
    let claim_rows: i64 = sp1_zkvm::io::read::<i64>();
    let claim_ncol: i64 = sp1_zkvm::io::read::<i64>();
    let claim_total: i64 = sp1_zkvm::io::read::<i64>();
    let claim_rank: i64 = sp1_zkvm::io::read::<i64>();
    assert_eq!(rows, claim_rows);
    assert_eq!(ncol, claim_ncol);
    assert_eq!(col_keep, claim_total);
    assert_eq!(2, claim_rank);
}
