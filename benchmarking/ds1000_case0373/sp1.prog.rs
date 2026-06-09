// DS-1000 case 373 — SP1 guest program.
// Kernel: ecdf_result(x) where xs = sort(x); ys = arange(1, len+1)/len; returns ys.
// For len = 50 sorted (any order), ys is deterministic: i/50 for i in 1..=50.
// Mirror the kernel: sort grades, then compute ys[i] = (i+1)/50.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let n: usize = 50;
    let mut x: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(sp1_zkvm::io::read::<f64>());
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        claimed.push(sp1_zkvm::io::read::<f64>());
    }

    // Sort via simple insertion sort (the values themselves are unused; only len matters,
    // but we still execute the sort to mirror the kernel's IR.)
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
