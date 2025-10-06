// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read a (2x2)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(env::read::<i32>());
        }
        a.push(row);
    }

    // read result (2x2)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(env::read::<i32>());
        }
        result.push(row);
    }

    // Step 1: compute min value over a
    let mut min_val: i32 = a[0][0];
    for i in 0..2 {
        for j in 0..2 {
            let val = a[i as usize][j as usize];
            if val < min_val {
                min_val = val;
            }
        }
    }

    // Step 2: collect indices where a[i,j] == min_val in row-major order
    let mut expected: Vec<Vec<i32>> = vec![vec![0; 2]; 2];
    let mut idx: usize = 0;
    for i in 0..2 {
        for j in 0..2 {
            if a[i as usize][j as usize] == min_val {
                expected[idx as usize][0] = i as i32;
                expected[idx as usize][1] = j as i32;
                idx += 1;
            }
        }
    }

    // Step 3: compare expected with result
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result[i as usize][j as usize], expected[i as usize][j as usize]);
        }
    }

    // env::commit(&output);
}
