// DS-1000 case 473 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut c: [i64; 8] = [0; 8];
    for k in 0..8 {
        c[k] = env::read();
    }
    let mut cnts: [i64; 3] = [0; 3];
    for k in 0..3 {
        cnts[k] = env::read();
    }
    let claimed: u64 = env::read();

    let mut result: bool = false;
    for x_idx in 0..3 {
        let mut all_match = true;
        for k in 0..8 {
            if c[k] != cnts[x_idx] {
                all_match = false;
            }
        }
        let shape_equal = false;
        result = result || (all_match && shape_equal);
    }

    let computed: u64 = if result { 1 } else { 0 };
    assert_eq!(computed, claimed);
}
