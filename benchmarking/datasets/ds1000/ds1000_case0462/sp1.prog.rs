// DS-1000 case 462 — SP1 guest.
// Kernel: keep a[i] iff (i == 0 or a[i] != a[i-1]) AND a[i] != 0.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [i64; 20] = [0; 20];
    for k in 0..20 { a[k] = sp1_zkvm::io::read::<i64>(); }
    let claimed_len: u64 = sp1_zkvm::io::read::<u64>();
    let mut claimed: Vec<i64> = Vec::with_capacity(claimed_len as usize);
    for _ in 0..claimed_len { claimed.push(sp1_zkvm::io::read::<i64>()); }

    let mut selection: [u64; 20] = [1; 20];
    for i in 1..20 {
        selection[i] = if a[i] != a[i-1] { 1 } else { 0 };
    }
    for i in 0..20 {
        let nz: u64 = if a[i] != 0 { 1 } else { 0 };
        selection[i] = selection[i] & nz;
    }

    let mut out: Vec<i64> = Vec::with_capacity(20);
    for i in 0..20 {
        if selection[i] == 1 {
            out.push(a[i]);
        }
    }

    assert_eq!(out.len() as u64, claimed_len);
    for k in 0..out.len() {
        assert_eq!(out[k], claimed[k]);
    }
}
