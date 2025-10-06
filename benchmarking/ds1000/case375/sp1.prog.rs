// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let n: usize = 27;

    // read grades (27)
    let mut grades: Vec<f64> = Vec::new();
    for _ in 0..n {
        grades.push(sp1_zkvm::io::read::<f64>());
    }

    // read threshold, low, high (scalars)
    let threshold: f64 = sp1_zkvm::io::read::<f64>();
    let low: f64 = sp1_zkvm::io::read::<f64>();
    let high: f64 = sp1_zkvm::io::read::<f64>();

    // 1) Verify sortedness (non-decreasing)
    for i in 0..(n - 1) {
        assert!(grades[i as usize] <= grades[(i + 1) as usize]);
    }

    // 2) Compute first index t where (k+1)/n > threshold
    let mut t: usize = n; // sentinel
    for k in 0..n {
        let cond: bool = (((k + 1) as f64) / (n as f64)) > threshold;
        if cond && (t == n) {
            t = k;
        }
    }

    // 3) Determine low, high
    let computed_low: f64 = grades[0];
    let computed_high: f64 = grades[t];

    // 4) Verify outputs
    assert!(low == computed_low);
    assert!(high == computed_high);

    // sp1_zkvm::io::commit_slice(&output);
}
