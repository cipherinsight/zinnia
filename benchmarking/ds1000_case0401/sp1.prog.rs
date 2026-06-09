// DS-1000 case 401 — SP1 guest program.
// Kernel: `np.array([[], [], []])` — produces a 3×0 empty array. No work.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // 3 empty rows; nothing to compute or assert.
    let _result: Vec<Vec<f64>> = vec![Vec::new(), Vec::new(), Vec::new()];
}
