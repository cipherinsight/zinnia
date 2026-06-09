// DS-1000 case 481 — SP1 guest program.
// kernel.py:
//     idx_list = ((x == a) & (y == b))
//     result = idx_list.nonzero()[0]
// Dynamic-length output: pass claimed length + padded buffer of size 9.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 9;
    let mut x: [i64; N] = [0; N];
    for k in 0..N { x[k] = sp1_zkvm::io::read::<i64>(); }
    let mut y: [i64; N] = [0; N];
    for k in 0..N { y[k] = sp1_zkvm::io::read::<i64>(); }
    let a: i64 = sp1_zkvm::io::read::<i64>();
    let b: i64 = sp1_zkvm::io::read::<i64>();
    let claimed_len: u64 = sp1_zkvm::io::read::<u64>();
    let mut claimed: [u64; N] = [0; N];
    for k in 0..N { claimed[k] = sp1_zkvm::io::read::<u64>(); }

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
