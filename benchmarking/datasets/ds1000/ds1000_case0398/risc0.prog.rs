// DS-1000 case 398 — Risc0 guest. 2-tap IIR recurrence.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 8;
    let mut a_arr: [f64; N] = [0.0; N];
    for k in 0..N { a_arr[k] = env::read(); }
    let a: f64 = env::read();
    let b: f64 = env::read();
    let mut claimed: [f64; N] = [0.0; N];
    for k in 0..N { claimed[k] = env::read(); }

    let mut b_arr: [f64; N] = [0.0; N];
    for k in 0..N {
        if k == 0 {
            b_arr[k] = a * a_arr[k];
        } else {
            b_arr[k] = a * a_arr[k] + b * b_arr[k - 1];
        }
    }

    for k in 0..N {
        assert!((b_arr[k] - claimed[k]).abs() < 0.001);
    }
}
