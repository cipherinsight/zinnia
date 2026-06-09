// DS-1000 case 463 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 20;
    let mut a: [i64; N] = [0; N];
    for k in 0..N {
        a[k] = env::read();
    }
    let claimed_len: u64 = env::read();
    let mut claimed: [i64; N] = [0; N];
    for k in 0..N {
        claimed[k] = env::read();
    }

    let mut selection: [bool; N] = [false; N];
    selection[0] = true;
    for k in 1..N {
        selection[k] = a[k] != a[k - 1];
    }
    for k in 0..N {
        selection[k] = selection[k] && (a[k] != 0);
    }

    let mut out: [i64; N] = [0; N];
    let mut out_len: u64 = 0;
    for k in 0..N {
        if selection[k] {
            out[out_len as usize] = a[k];
            out_len += 1;
        }
    }

    assert_eq!(out_len, claimed_len);
    for k in 0..(claimed_len as usize) {
        assert_eq!(out[k], claimed[k]);
    }
}
