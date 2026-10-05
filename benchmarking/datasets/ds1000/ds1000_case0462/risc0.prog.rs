// DS-1000 case 462 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [i64; 20] = [0; 20];
    for k in 0..20 { a[k] = env::read(); }
    let claimed_len: u64 = env::read();
    let mut claimed: Vec<i64> = Vec::with_capacity(claimed_len as usize);
    for _ in 0..claimed_len { claimed.push(env::read()); }

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
