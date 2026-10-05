// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read input array
    let mut a: Vec<f32> = Vec::new();
    for _ in 0..13 {
        a.push(env::read());
    }

    // Read result flags
    let mut result: Vec<u32> = Vec::new();
    for _ in 0..13 {
        result.push(env::read());
    }

    let n: f32 = 13.0;

    // Compute mean
    let mut sum = 0.0;
    for i in 0..13 {
        sum += a[i];
    }
    let mean_val = sum / n;

    // Compute variance
    let mut var_sum = 0.0;
    for i in 0..13 {
        let diff = a[i] - mean_val;
        var_sum += diff * diff;
    }
    let variance = var_sum / n;
    let std_val = variance.sqrt();

    let lower = mean_val - 2.0 * std_val;
    let upper = mean_val + 2.0 * std_val;

    // Verify boolean flags
    for i in 0..13 {
        let inside = a[i] > lower && a[i] < upper;
        let expected: u32 = if !inside { 1 } else { 0 };
        assert_eq!(result[i], expected);
    }

    // env::commit(&output);
}
