// DS-1000 case 381 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let n: usize = 100;
    let mut a: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        a.push(env::read());
    }
    let mut claimed: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        claimed.push(env::read());
    }

    let mut idx: Vec<usize> = (0..n).collect();
    for i in 1..n {
        let key = idx[i];
        let mut j: usize = i;
        while j > 0 && a[idx[j - 1]] > a[key] {
            idx[j] = idx[j - 1];
            j -= 1;
        }
        idx[j] = key;
    }
    let mut rev: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        rev.push(idx[n - 1 - i] as u64);
    }

    for i in 0..n {
        assert_eq!(rev[i], claimed[i]);
    }
}
