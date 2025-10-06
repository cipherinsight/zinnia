// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read input a (3x2x2)
    let mut a: Vec<Vec<Vec<i32>>> = Vec::new();
    for _ in 0..3 {
        let mut mat: Vec<Vec<i32>> = Vec::new();
        for _ in 0..2 {
            let mut row: Vec<i32> = Vec::new();
            for _ in 0..2 {
                row.push(env::read());
            }
            mat.push(row);
        }
        a.push(mat);
    }

    // read permutation (3)
    let mut permutation: Vec<i32> = Vec::new();
    for _ in 0..3 {
        permutation.push(env::read());
    }

    // read result (3x2x2)
    let mut result: Vec<Vec<Vec<i32>>> = Vec::new();
    for _ in 0..3 {
        let mut mat: Vec<Vec<i32>> = Vec::new();
        for _ in 0..2 {
            let mut row: Vec<i32> = Vec::new();
            for _ in 0..2 {
                row.push(env::read());
            }
            mat.push(row);
        }
        result.push(mat);
    }

    // Logic: result[k,r,s] == a[c[k],r,s] with c[k] = sum_i i*[permutation[i]==k]
    for k in 0..3 {
        let mut ck: i32 = 0;
        for i in 0..3 {
            let is_target: i32 = if permutation[i as usize] == k as i32 { 1 } else { 0 };
            ck += (i as i32) * is_target;
        }

        for r in 0..2 {
            for s in 0..2 {
                let mut selected: i32 = 0;
                for t in 0..3 {
                    let ind: i32 = if ck == t as i32 { 1 } else { 0 };
                    selected += a[t as usize][r as usize][s as usize] * ind;
                }
                assert_eq!(result[k as usize][r as usize][s as usize], selected);
            }
        }
    }

    // env::commit(&output);
}
