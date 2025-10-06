// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read x (13)
    let mut x: Vec<f32> = Vec::new();
    for _ in 0..13 {
        x.push(env::read());
    }

    // read result (10)
    let mut result: Vec<f32> = Vec::new();
    for _ in 0..10 {
        result.push(env::read());
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

    // env::commit(&output);
}
