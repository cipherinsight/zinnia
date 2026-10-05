// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read inputs
    let mut a: Vec<f32> = Vec::new();
    for _ in 0..5 {
        a.push(sp1_zkvm::io::read::<f32>());
    }
    let p: f32 = sp1_zkvm::io::read::<f32>();
    let result: f32 = sp1_zkvm::io::read::<f32>();

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

    // sp1_zkvm::io::commit_slice(&output);
}
