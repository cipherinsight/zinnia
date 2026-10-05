// DS-1000 case 0489 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 8;
    let mut arr: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N { arr.push(env::read()); }
    let result: Vec<f64> = arr.clone();
    for i in 0..N {
        if result[i] < -10.0 {
            arr[i] = 0.0;
        } else if result[i] >= 15.0 {
            arr[i] = 30.0;
        } else {
            arr[i] += 5.0;
        }
    }
    for i in 0..N {
        let claim: f64 = env::read();
        assert!((arr[i] - claim).abs() < 1e-9);
    }
}
