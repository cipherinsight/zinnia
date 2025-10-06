// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read input a (2x3)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        a.push(row);
    }

    // read input result
    let result: i32 = sp1_zkvm::io::read::<i32>();

    // Flatten in C order (row-major)
    let mut flat: Vec<i32> = Vec::new();
    for i in 0..2 {
        for j in 0..3 {
            flat.push(a[i][j as usize]);
        }
    }

    // Compute argmin
    let mut min_val: i32 = flat[0];
    let mut min_idx: usize = 0;
    for i in 1..flat.len() {
        if flat[i] < min_val {
            min_val = flat[i];
            min_idx = i;
        }
    }

    // Check equality
    assert_eq!(result, min_idx as i32);

    // sp1_zkvm::io::commit_slice(&output);
}
