// DS-1000 case 373 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let n: usize = 50;
    let mut x: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(env::read());
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        claimed.push(env::read());
    }

    let mut xs: Vec<f64> = x.clone();
    for i in 1..n {
        let key = xs[i];
        let mut j: usize = i;
        while j > 0 && xs[j - 1] > key {
            xs[j] = xs[j - 1];
            j -= 1;
        }
        xs[j] = key;
    }

    let nf: f64 = n as f64;
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        ys.push((i as f64 + 1.0) / nf);
    }

    let _ = &xs;

    for i in 0..n {
        assert!((ys[i] - claimed[i]).abs() < 1e-9);
    }
}
