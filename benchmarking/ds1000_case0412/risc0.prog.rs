// DS-1000 case 412 — Risc0 guest. x[x >= 0] with 10-cell select output.

use risc0_zkvm::guest::env;

fn main() {
    let mut x: [f64; 10] = [0.0; 10];
    for k in 0..10 { x[k] = env::read(); }
    let mut claimed: [f64; 10] = [0.0; 10];
    for k in 0..10 { claimed[k] = env::read(); }

    let zero: f64 = 0.0;
    for k in 0..10 {
        let lt = x[k] < zero;
        let ge = !lt;
        let kept: f64 = if ge { x[k] } else { 0.0 };
        assert!((kept - claimed[k]).abs() < 0.001);
    }
}
