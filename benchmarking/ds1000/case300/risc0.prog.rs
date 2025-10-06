// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read inputs
    let mut a: Vec<f32> = Vec::new();
    for _ in 0..5 {
        a.push(env::read());
    }
    let p: f32 = env::read();
    let result: f32 = env::read();

    let n: usize = 5;

    // verify sorted order
    for i in 0..(n - 1) {
        assert!(a[i] <= a[i + 1]);
    }

    // compute percentile rank
    let rank: f32 = (p / 100.0) * ((n - 1) as f32);
    let lower: usize = rank.floor() as usize;
    let upper: usize = lower + 1;
    let fraction: f32 = rank - (lower as f32);

    // interpolation
    let interpolated: f32 = a[lower] + (a[upper] - a[lower]) * fraction;

    assert!((result - interpolated).abs() < 1e-6);

    // env::commit(&output);
}
