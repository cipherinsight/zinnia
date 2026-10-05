// DS-1000 case 375 — SP1 guest program.
// Kernel: sort grades into xs; ys = arange(1, N+1)/N; t = (ys > threshold).argmax();
// low = xs[0]; high = xs[t]; return grades (unchanged).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let n: usize = 50;
    let mut grades: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        grades.push(sp1_zkvm::io::read::<f64>());
    }
    let threshold: f64 = sp1_zkvm::io::read::<f64>();
    let mut claimed: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        claimed.push(sp1_zkvm::io::read::<f64>());
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

    // t = (ys > threshold).argmax() — index of first ys[k] > threshold.
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
    // low/high are computed but discarded; kernel returns grades.
    let _ = (low, high);

    // Return grades unchanged.
    for i in 0..n {
        assert!((grades[i] - claimed[i]).abs() < 1e-9);
    }
}
