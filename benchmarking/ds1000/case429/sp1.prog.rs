// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read array
    let mut a: Vec<f32> = Vec::new();
    for _ in 0..13 {
        a.push(sp1_zkvm::io::read::<f32>());
    }

    // read result (tuple of two floats)
    let lower_res: f32 = sp1_zkvm::io::read::<f32>();
    let upper_res: f32 = sp1_zkvm::io::read::<f32>();

    let n: f32 = 13.0;

    // compute mean
    let mut sum = 0.0;
    for i in 0..13 {
        sum += a[i];
    }
    let mean_val = sum / n;

    // compute variance
    let mut var_sum = 0.0;
    for i in 0..13 {
        let diff = a[i] - mean_val;
        var_sum += diff * diff;
    }
    let variance = var_sum / n;
    let std_val = variance.sqrt();

    let lower = mean_val - 2.0 * std_val;
    let upper = mean_val + 2.0 * std_val;

    assert!((lower - lower_res).abs() < 1e-6);
    assert!((upper - upper_res).abs() < 1e-6);

    // sp1_zkvm::io::commit_slice(&output);
}
