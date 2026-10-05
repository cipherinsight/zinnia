// DS-1000 case 374 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let n: usize = 50;
    let m: usize = 5;
    let mut grades: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        grades.push(env::read());
    }
    let mut eval: Vec<i64> = Vec::with_capacity(m);
    for _ in 0..m {
        eval.push(env::read());
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(m);
    for _ in 0..m {
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

    let mut result: Vec<f64> = vec![0.0; m];
    for i in 0..m {
        let element: f64 = eval[i] as f64;
        if element < xs[0] {
            result[i] = 0.0;
        } else if element >= xs[n - 1] {
            result[i] = 1.0;
        } else {
            let mut best_idx: usize = 0;
            let mut found: bool = false;
            for k in 0..n {
                if !found && xs[k] > element {
                    best_idx = k;
                    found = true;
                }
            }
            result[i] = ys[best_idx - 1];
        }
    }

    for i in 0..m {
        assert!((result[i] - claimed[i]).abs() < 1e-9);
    }
}
