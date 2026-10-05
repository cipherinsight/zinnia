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

    // read result (27)
    let mut result: Vec<f64> = Vec::new();
    for _ in 0..n {
        result.push(sp1_zkvm::io::read::<f64>());
    }

    // 1) Validate sortedness (non-decreasing)
    for i in 0..(n - 1) {
        assert!(grades[i as usize] <= grades[(i + 1) as usize]);
    }

    // 2) ECDF values at sorted sample points: i/n for i=1..n
    let mut ys: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        ys[i as usize] = ((i + 1) as f64) / (n as f64);
    }

    // 3) Verify output elementwise equality
    for i in 0..n {
        assert!(result[i as usize] == ys[i as usize]);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
