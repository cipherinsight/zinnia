// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read input a (2x2)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        a.push(row);
    }

    // read input result (2x2)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        result.push(row);
    }

    // Step 1: Compute min value
    let mut min_val: i32 = a[0][0];
    for i in 0..2 {
        for j in 0..2 {
            if a[i as usize][j as usize] < min_val {
                min_val = a[i as usize][j as usize];
            }
        }
    }

    // Step 2: Build expected matrix
    let mut expected: Vec<Vec<i32>> = vec![vec![0; 2], vec![0; 2]];
    let mut idx: usize = 0;
    for i in 0..2 {
        for j in 0..2 {
            if a[i as usize][j as usize] == min_val {
                expected[idx][0] = i as i32;
                expected[idx][1] = j as i32;
                idx += 1;
            }
        }
    }

    // Step 3: Compare with result
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result[i as usize][j as usize], expected[i as usize][j as usize]);
        }
    }

    // sp1_zkvm::io::commit_slice(&output);
}
