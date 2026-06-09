// DS-1000 case 460 — SP1 guest.
// Kernel: `np.mean(NA.astype(float), axis=0)` returning a single mean.
// A: Float[5] is witnessed but unused (mirrors Zinnia IR).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [f64; 5] = [0.0; 5];
    for k in 0..5 { a[k] = sp1_zkvm::io::read::<f64>(); }
    let _a = a; // explicit no-op consumer

    let mut na: [i64; 5] = [0; 5];
    for k in 0..5 { na[k] = sp1_zkvm::io::read::<i64>(); }
    let claimed: f64 = sp1_zkvm::io::read::<f64>();

    let mut na_f: [f64; 5] = [0.0; 5];
    for k in 0..5 { na_f[k] = na[k] as f64; }

    let mut sum: f64 = na_f[0];
    for k in 1..5 { sum = sum + na_f[k]; }
    let n: f64 = 5.0;
    let mean: f64 = sum / n;

    assert!((mean - claimed).abs() < 0.001);
}
