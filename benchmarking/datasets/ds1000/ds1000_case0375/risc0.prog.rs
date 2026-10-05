// DS-1000 case 375 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let n: usize = 50;
    let mut grades: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        grades.push(env::read());
    }
    let threshold: f64 = env::read();
    let mut claimed: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        claimed.push(env::read());
    }

    let mut xs: Vec<f64> = grades.clone();
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

    let mut t: usize = 0;
    let mut found: bool = false;
    for k in 0..n {
        if !found && ys[k] > threshold {
            t = k;
            found = true;
        }
    }
    let low: f64 = xs[0];
    let high: f64 = xs[t];
    let _ = (low, high);

    for i in 0..n {
        assert!((grades[i] - claimed[i]).abs() < 1e-9);
    }
}
