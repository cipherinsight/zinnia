// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read a, b, c (3-element float arrays)
    let mut a: Vec<f64> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut c: Vec<f64> = Vec::new();
    for _ in 0..3 { a.push(env::read()); }
    for _ in 0..3 { b.push(env::read()); }
    for _ in 0..3 { c.push(env::read()); }

    // read result (3-element float array)
    let mut result: Vec<f64> = Vec::new();
    for _ in 0..3 { result.push(env::read()); }

    // compute mean over a,b,c (axis=0)
    let mut computed: Vec<f64> = Vec::new();
    for i in 0..3 {
        let mean_val = (a[i as usize] + b[i as usize] + c[i as usize]) / 3.0;
        computed.push(mean_val);
    }

    // compare result == computed (with tolerance)
    for i in 0..3 {
        assert!((result[i as usize] - computed[i as usize]).abs() < 1e-9);
    }

    // env::commit(&output);
}
