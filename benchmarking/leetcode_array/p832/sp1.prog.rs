//! A simple program that takes a number `n` as input, and writes the `n-1`th and `n`th fibonacci
//! number as an output.

// These two lines are necessary for the program to properly compile.
//
// Under the hood, we wrap your main function with some extra code so that it behaves properly
// inside the zkVM.
#![no_main]
sp1_zkvm::entrypoint!(main);

use alloy_sol_types::SolType;
use fibonacci_lib::PublicValuesStruct;

pub fn main() {
    // Read an input to the program.
    //
    // Behind the scenes, this compiles down to a custom system call which handles reading inputs
    // from the prover.
    let mut image: Vec<Vec<u32>> = Vec::new();
    let mut result: Vec<Vec<u32>> = Vec::new();

    for i in 0..10 {
        let mut tmp = Vec::new();
        for j in 0..10 {
            let n = sp1_zkvm::io::read::<u32>();
            tmp.push(n);
        }
        image.push(tmp);
    }
    for i in 0..10 {
        let mut tmp = Vec::new();
        for j in 0..10 {
            let n = sp1_zkvm::io::read::<u32>();
            tmp.push(n);
        }
        result.push(tmp);
    }

    // Check that both image and result are 10x10 arrays.
    assert_eq!(image.len(), 10, "Image must have 10 rows");
    assert_eq!(result.len(), 10, "Result must have 10 rows");
    for (i, row) in image.iter().enumerate() {
        assert_eq!(row.len(), 10, "Row {} in image must have 10 columns", i);
        for (j, &val) in row.iter().enumerate() {
            assert!(
                val == 0 || val == 1,
                "Image element at ({},{}) must be 0 or 1",
                i,
                j
            );
        }
    }
    for (i, row) in result.iter().enumerate() {
        assert_eq!(row.len(), 10, "Row {} in result must have 10 columns", i);
        for (j, &val) in row.iter().enumerate() {
            assert!(
                val == 0 || val == 1,
                "Result element at ({},{}) must be 0 or 1",
                i,
                j
            );
        }
    }

    // For each element, verify that result[i][j] == 1 - image[i][9 - j]
    for i in 0..10 {
        for j in 0..10 {
            assert_eq!(
                result[i][j],
                1 - image[i][10 - 1 - j],
                "Mismatch at position ({},{}): expected {} but got {}",
                i,
                j,
                1 - image[i][10 - 1 - j],
                result[i][j]
            );
        }
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
