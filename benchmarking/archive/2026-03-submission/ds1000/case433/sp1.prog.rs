// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // Read input 4x4 matrix
    let mut a: [[i32; 4]; 4] = [[0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = sp1_zkvm::io::read::<i32>();
        }
    }

    // Read expected result
    let mut result: [[i32; 4]; 4] = [[0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = sp1_zkvm::io::read::<i32>();
        }
    }

    let zero_rows = 0;
    let zero_cols = 0;

    let mut modified = a;
    for j in 0..4 {
        modified[zero_rows][j] = 0;
    }
    for i in 0..4 {
        modified[i][zero_cols] = 0;
    }

    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(result[i][j], modified[i][j]);
        }
    }

    // sp1_zkvm::io::commit_slice(&output);
}
