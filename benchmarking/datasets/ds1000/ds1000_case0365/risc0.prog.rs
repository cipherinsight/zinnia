// DS-1000 case 365 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(100);
    for _ in 0..100 {
        a.push(env::read());
    }
    let pos: u64 = env::read();
    let element: i64 = env::read();
    let mut claimed: Vec<f64> = Vec::with_capacity(101);
    for _ in 0..101 {
        claimed.push(env::read());
    }

    let mut result: Vec<f64> = Vec::with_capacity(101);
    for i in 0..101usize {
        let v: f64 = if (i as u64) < pos {
            a[i]
        } else if (i as u64) == pos {
            element as f64
        } else {
            a[i - 1]
        };
        result.push(v);
    }

    for i in 0..101 {
        assert!((result[i] - claimed[i]).abs() < 1e-9);
    }
}
