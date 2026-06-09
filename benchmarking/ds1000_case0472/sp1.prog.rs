// DS-1000 case 472 — SP1 guest program.
// kernel.py: a[-1:, ...] on Float[6, 2, 9, 3, 7, 4].
// Returns the last slab along axis 0, shape (1, 2, 9, 3, 7, 4) = 1512 floats.
// Inputs are 9072 floats; the claimed output is 1512 floats matching the
// last 1512 of the input flattened in row-major order.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 6 * 2 * 9 * 3 * 7 * 4; // 9072
    const SLAB: usize = 2 * 9 * 3 * 7 * 4;  // 1512
    let mut a: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        a.push(sp1_zkvm::io::read::<f64>());
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(SLAB);
    for _ in 0..SLAB {
        claimed.push(sp1_zkvm::io::read::<f64>());
    }

    // Last slab starts at index 5 * SLAB.
    let start = 5 * SLAB;
    for k in 0..SLAB {
        let computed = a[start + k];
        assert!((computed - claimed[k]).abs() <= 0.001);
    }
}
