// DS-1000 case 402 — Risc0 guest. ravel_multi_index 'F'.

use risc0_zkvm::guest::env;

fn main() {
    let mut dims: [u64; 5] = [0; 5];
    for k in 0..5 { dims[k] = env::read(); }
    let total: usize = 8 * 9 * 8 * 8 * 8;
    let mut s: f64 = 0.0;
    for _ in 0..total {
        let v: f64 = env::read();
        s += v * 0.0;
    }
    let mut index: [u64; 5] = [0; 5];
    for k in 0..5 { index[k] = env::read(); }
    let claimed: u64 = env::read();

    let mut result: u64 = 0;
    let mut stride: u64 = 1;
    for k in 0..5 {
        result += index[k] * stride;
        stride *= dims[k];
    }

    assert_eq!(result, claimed);
    assert!(s == 0.0);
}
