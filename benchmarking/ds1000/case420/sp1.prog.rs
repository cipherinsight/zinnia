// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let x: f32 = sp1_zkvm::io::read::<f32>();
    let result: f32 = sp1_zkvm::io::read::<f32>();

    let x_min: f32 = 0.0;
    let x_max: f32 = 1.0;

    let mut expected: f32 = x_min;
    if x > x_max {
        expected = x_max;
    } else if x >= x_min {
        expected = 3.0 * x * x - 2.0 * x * x * x;
    }

    assert!((result - expected).abs() < 1e-6);

    // sp1_zkvm::io::commit_slice(&output);
}
