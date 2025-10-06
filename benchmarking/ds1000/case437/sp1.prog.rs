// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // Read input 3×2 matrix
    let mut a: [[i32; 2]; 3] = [[0; 2]; 3];
    for i in 0..3 {
        for j in 0..2 {
            a[i][j] = sp1_zkvm::io::read::<i32>();
        }
    }

    // Read mask
    let mut mask: [[i32; 2]; 3] = [[0; 2]; 3];
    for i in 0..3 {
        for j in 0..2 {
            mask[i][j] = sp1_zkvm::io::read::<i32>();
        }
    }

    // Compute expected mask
    for i in 0..3 {
        let mut row_min = a[i][0];
        for j in 1..2 {
            if a[i][j] < row_min {
                row_min = a[i][j];
            }
        }
        for j in 0..2 {
            let expected = if a[i][j] == row_min { 1 } else { 0 };
            assert_eq!(mask[i][j], expected);
        }
    }

    // sp1_zkvm::io::commit_slice(&output);
}
