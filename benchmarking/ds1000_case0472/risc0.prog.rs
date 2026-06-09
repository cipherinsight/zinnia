// DS-1000 case 472 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 9072;
    const SLAB: usize = 1512;
    let mut a: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        a.push(env::read());
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(SLAB);
    for _ in 0..SLAB {
        claimed.push(env::read());
    }

    let start = 5 * SLAB;
    for k in 0..SLAB {
        let computed = a[start + k];
        assert!((computed - claimed[k]).abs() <= 0.001);
    }
}
