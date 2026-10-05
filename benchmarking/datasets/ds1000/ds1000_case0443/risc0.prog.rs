// DS-1000 case 443 — Risc0 guest.
// Kernel: `C = A[np.in1d(A, B)]`.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [i64; 20] = [0; 20];
    for k in 0..20 { a[k] = env::read(); }
    let mut b: [i64; 3] = [0; 3];
    for k in 0..3 { b[k] = env::read(); }
    let claimed_len: u64 = env::read();
    let mut claimed: Vec<i64> = Vec::with_capacity(claimed_len as usize);
    for _ in 0..claimed_len { claimed.push(env::read()); }

    let mut out: Vec<i64> = Vec::with_capacity(20);
    for i in 0..20 {
        let mut in_b: u64 = 0;
        for j in 0..3 {
            let eq: u64 = if a[i] == b[j] { 1 } else { 0 };
            in_b = in_b | eq;
        }
        if in_b == 1 {
            out.push(a[i]);
        }
    }

    assert_eq!(out.len() as u64, claimed_len);
    for k in 0..out.len() {
        assert_eq!(out[k], claimed[k]);
    }
}
