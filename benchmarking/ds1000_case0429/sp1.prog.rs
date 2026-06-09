// DS-1000 case 429 — SP1 guest.
// Kernel: (a.mean() - 2*a.std(), a.mean() + 2*a.std()) on Float[30].
// SqrtFIR emitted twice; mirrored.

#![no_main]
sp1_zkvm::entrypoint!(main);

fn std_of(a: &[f64], mean: f64) -> f64 {
    let n = a.len() as f64;
    let mut s: f64 = 0.0;
    for v in a.iter() {
        let d = *v - mean;
        s += d * d;
    }
    let var = s / n;
    var.sqrt()
}

pub fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(30);
    for _ in 0..30 { a.push(sp1_zkvm::io::read::<f64>()); }
    let claimed_lo: f64 = sp1_zkvm::io::read::<f64>();
    let claimed_hi: f64 = sp1_zkvm::io::read::<f64>();

    let mut sum: f64 = 0.0;
    for v in a.iter() { sum += *v; }
    let mean = sum / 30.0;

    let std1 = std_of(&a, mean);
    let std2 = std_of(&a, mean);

    let lo = mean - 2.0 * std1;
    let hi = mean + 2.0 * std2;

    assert!((lo - claimed_lo).abs() < 0.005);
    assert!((hi - claimed_hi).abs() < 0.005);
}
