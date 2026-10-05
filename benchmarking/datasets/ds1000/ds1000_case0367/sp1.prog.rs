// DS-1000 case 367 — SP1 guest program.
// Kernel: `copy.deepcopy(array_of_arrays)`. `pairs` is unused (still read to mirror IR).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut pairs: Vec<Vec<i64>> = Vec::with_capacity(3);
    for _ in 0..3 {
        let mut row: Vec<i64> = Vec::with_capacity(2);
        for _ in 0..2 {
            row.push(sp1_zkvm::io::read::<i64>());
        }
        pairs.push(row);
    }
    let mut array_of_arrays: Vec<i64> = Vec::with_capacity(3);
    for _ in 0..3 {
        array_of_arrays.push(sp1_zkvm::io::read::<i64>());
    }
    let mut claimed: Vec<i64> = Vec::with_capacity(3);
    for _ in 0..3 {
        claimed.push(sp1_zkvm::io::read::<i64>());
    }

    // Deep copy: produce an independent vector with the same values.
    let mut result: Vec<i64> = Vec::with_capacity(3);
    for i in 0..3 {
        result.push(array_of_arrays[i]);
    }

    // Suppress unused warning while preserving the IR shape.
    let _ = &pairs;

    for i in 0..3 {
        assert_eq!(result[i], claimed[i]);
    }
}
