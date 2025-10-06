// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read x (13)
    let mut x: Vec<f32> = Vec::new();
    for _ in 0..13 {
        x.push(sp1_zkvm::io::read::<f32>());
    }

    // read result (10)
    let mut result: Vec<f32> = Vec::new();
    for _ in 0..10 {
        result.push(sp1_zkvm::io::read::<f32>());
    }

    // filter x >= 0
    let mut filtered: Vec<f32> = Vec::new();
    for i in 0..x.len() {
        if x[i] >= 0.0 {
            filtered.push(x[i]);
        }
    }

    let expected = filtered;

    for i in 0..result.len() {
        assert!((result[i] - expected[i]).abs() < 1e-6);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
