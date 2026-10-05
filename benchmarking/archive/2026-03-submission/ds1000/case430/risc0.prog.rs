// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read array
    let mut a: Vec<f32> = Vec::new();
    for _ in 0..13 {
        a.push(env::read());
    }

    // Read result (tuple)
    let lower_res: f32 = env::read();
    let upper_res: f32 = env::read();

    let n: f32 = 13.0;

    // Mean
    let mut sum = 0.0;
    for i in 0..13 {
        sum += a[i];
    }
    let mean_val = sum / n;

    // Variance
    let mut var_sum = 0.0;
    for i in 0..13 {
        let diff = a[i] - mean_val;
        var_sum += diff * diff;
    }
    let variance = var_sum / n;
    let std_val = variance.sqrt();

    let lower = mean_val - 3.0 * std_val;
    let upper = mean_val + 3.0 * std_val;

    assert!((lower - lower_res).abs() < 1e-6);
    assert!((upper - upper_res).abs() < 1e-6);

    // env::commit(&output);
}
