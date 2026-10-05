// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read input a (2x5)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..5 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        a.push(row);
    }

    // read permutation (5)
    let mut permutation: Vec<i32> = Vec::new();
    for _ in 0..5 {
        permutation.push(sp1_zkvm::io::read::<i32>());
    }

    // read result (2x5)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..5 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        result.push(row);
    }

    for j in 0..5 {
        // compute c[j]
        let mut cj: i32 = 0;
        for i in 0..5 {
            let is_target: i32 = if permutation[i as usize] == j as i32 { 1 } else { 0 };
            cj = cj + (i as i32) * is_target;
        }

        // select for row 0
        let mut sel_val_r0: i32 = 0;
        for t in 0..5 {
            let ind: i32 = if cj == t as i32 { 1 } else { 0 };
            sel_val_r0 = sel_val_r0 + a[0][t as usize] * ind;
        }
        assert_eq!(result[0][j as usize], sel_val_r0);

        // select for row 1
        let mut sel_val_r1: i32 = 0;
        for t in 0..5 {
            let ind: i32 = if cj == t as i32 { 1 } else { 0 };
            sel_val_r1 = sel_val_r1 + a[1][t as usize] * ind;
        }
        assert_eq!(result[1][j as usize], sel_val_r1);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
