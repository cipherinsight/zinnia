// DS-1000 case 444 — Risc0 guest.

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
        let v = a[i];
        let lo1: u64 = if v > b[0] { 1 } else { 0 };
        let hi1: u64 = if v < b[1] { 1 } else { 0 };
        let lo2: u64 = if v > b[1] { 1 } else { 0 };
        let hi2: u64 = if v < b[2] { 1 } else { 0 };
        let band1: u64 = lo1 & hi1;
        let band2: u64 = lo2 & hi2;
        let mask: u64 = band1 | band2;
        if mask == 1 {
            out.push(v);
        }
    }

    assert_eq!(out.len() as u64, claimed_len);
    for k in 0..out.len() {
        assert_eq!(out[k], claimed[k]);
    }
}
