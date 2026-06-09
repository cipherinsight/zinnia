// DS-1000 case 399 — Risc0 guest. 3-tap IIR with array tap c.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 8;
    let mut a_arr: [f64; N] = [0.0; N];
    for k in 0..N { a_arr[k] = env::read(); }
    let a: f64 = env::read();
    let b: f64 = env::read();
    let mut c_arr: [f64; N] = [0.0; N];
    for k in 0..N { c_arr[k] = env::read(); }
    let mut claimed: [f64; N] = [0.0; N];
    for k in 0..N { claimed[k] = env::read(); }

    let mut b_arr: [f64; N] = [0.0; N];
    for k in 0..N {
        if k == 0 {
            b_arr[k] = a * a_arr[k];
        } else if k == 1 {
            b_arr[k] = a * a_arr[k] + b * b_arr[k - 1];
        } else {
            b_arr[k] = a * a_arr[k] + b * b_arr[k - 1] + c_arr[k] * b_arr[k - 2];
        }
    }

    for k in 0..N {
        assert!((b_arr[k] - claimed[k]).abs() < 0.001);
    }
}
