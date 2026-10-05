// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // Read float input array
    let mut a: Vec<f32> = Vec::new();
    for _ in 0..13 {
        a.push(sp1_zkvm::io::read::<f32>());
    }

    // Read result flags (as u32)
    let mut result: Vec<u32> = Vec::new();
    for _ in 0..13 {
        result.push(sp1_zkvm::io::read::<u32>());
    }

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

    let lower = mean_val - 2.0 * std_val;
    let upper = mean_val + 2.0 * std_val;

    for i in 0..13 {
        let inside = a[i] > lower && a[i] < upper;
        let expected: u32 = if !inside { 1 } else { 0 };
        assert_eq!(result[i], expected);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
