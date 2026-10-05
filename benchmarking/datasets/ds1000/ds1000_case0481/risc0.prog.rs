// DS-1000 case 481 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 9;
    let mut x: [i64; N] = [0; N];
    for k in 0..N { x[k] = env::read(); }
    let mut y: [i64; N] = [0; N];
    for k in 0..N { y[k] = env::read(); }
    let a: i64 = env::read();
    let b: i64 = env::read();
    let claimed_len: u64 = env::read();
    let mut claimed: [u64; N] = [0; N];
    for k in 0..N { claimed[k] = env::read(); }

    let mut idx_list: [bool; N] = [false; N];
    for k in 0..N {
        idx_list[k] = (x[k] == a) && (y[k] == b);
    }

    let mut out: [u64; N] = [0; N];
    let mut out_len: u64 = 0;
    for k in 0..N {
        if idx_list[k] {
            out[out_len as usize] = k as u64;
            out_len += 1;
        }
    }

    assert_eq!(out_len, claimed_len);
    for k in 0..(claimed_len as usize) {
        assert_eq!(out[k], claimed[k]);
    }
}
