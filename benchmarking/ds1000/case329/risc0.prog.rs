// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read input a (2x2)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(env::read());
        }
        a.push(row);
    }

    // read power
    let power: i32 = env::read();

    // read result (2x2)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(env::read());
        }
        result.push(row);
    }

    // compute elementwise power
    let mut computed: Vec<Vec<i32>> = vec![vec![0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut val: i32 = 1;
            for _ in 0..power {
                val *= a[i as usize][j as usize];
            }
            computed[i as usize][j as usize] = val;
        }
    }

    // compare with result
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result[i as usize][j as usize], computed[i as usize][j as usize]);
        }
    }

    // env::commit(&output);
}
