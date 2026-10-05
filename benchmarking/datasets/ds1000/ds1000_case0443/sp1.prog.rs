// DS-1000 case 443 — SP1 guest.
// Kernel: `C = A[np.in1d(A, B)]` on Integer[20] × Integer[3].
// Returns elements of A that appear in B.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [i64; 20] = [0; 20];
    for k in 0..20 { a[k] = sp1_zkvm::io::read::<i64>(); }
    let mut b: [i64; 3] = [0; 3];
    for k in 0..3 { b[k] = sp1_zkvm::io::read::<i64>(); }
    let claimed_len: u64 = sp1_zkvm::io::read::<u64>();
    let mut claimed: Vec<i64> = Vec::with_capacity(claimed_len as usize);
    for _ in 0..claimed_len { claimed.push(sp1_zkvm::io::read::<i64>()); }

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
