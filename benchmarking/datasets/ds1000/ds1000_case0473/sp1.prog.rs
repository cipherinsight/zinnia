// DS-1000 case 473 — SP1 guest program.
// kernel.py: result = any(np.array_equal(c, x) for x in CNTS)
// with c: Integer[4, 1, 2] and CNTS: Integer[3].
// np.array_equal between an 8-element array and a scalar is always False
// (shape mismatch), so the result is always False. We still iterate through
// CNTS and through every element of c to keep the work shape comparable.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut c: [i64; 8] = [0; 8];
    for k in 0..8 {
        c[k] = sp1_zkvm::io::read::<i64>();
    }
    let mut cnts: [i64; 3] = [0; 3];
    for k in 0..3 {
        cnts[k] = sp1_zkvm::io::read::<i64>();
    }
    let claimed: u64 = sp1_zkvm::io::read::<u64>(); // 0 (false) or 1 (true)

    let mut result: bool = false;
    for x_idx in 0..3 {
        // Compare 8-element array with scalar — shape mismatch → False.
        // To preserve the structural work, we count matching elements and only
        // declare equality when count == 8 AND scalar comparison holds elementwise.
        let mut all_match = true;
        for k in 0..8 {
            if c[k] != cnts[x_idx] {
                all_match = false;
            }
        }
        // Even if all_match is true, numpy's array_equal returns False for
        // mismatched shapes; mirror that by ANDing with "shape equal" which is
        // always false here.
        let shape_equal = false;
        result = result || (all_match && shape_equal);
    }

    let computed: u64 = if result { 1 } else { 0 };
    assert_eq!(computed, claimed);
}
