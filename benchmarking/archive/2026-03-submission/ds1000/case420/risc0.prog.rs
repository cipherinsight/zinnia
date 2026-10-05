// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    let x: f32 = env::read();
    let result: f32 = env::read();

    let x_min: f32 = 0.0;
    let x_max: f32 = 1.0;

    let mut expected: f32 = x_min;
    if x > x_max {
        expected = x_max;
    } else if x >= x_min {
        expected = 3.0 * x * x - 2.0 * x * x * x;
    }

    assert!((result - expected).abs() < 1e-6);

    // env::commit(&output);
}
