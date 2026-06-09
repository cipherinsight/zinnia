// DS-1000 case 400 — Risc0 guest program.
// Kernel: `np.array([])` — empty array, no work.

use risc0_zkvm::guest::env;

fn main() {
    let _ = env::cycle_count(); // touch the env so the crate is linked
    let _result: Vec<f64> = Vec::new();
}
