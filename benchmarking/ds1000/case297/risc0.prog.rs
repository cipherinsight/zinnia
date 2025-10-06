// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read input arrays
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..3 {
        a.push(env::read());
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..5 {
            tmp.push(env::read());
        }
        result.push(tmp);
    }

    // compute a_min
    let mut a_min = a[0];
    for i in 1..3 {
        if a[i] < a_min {
            a_min = a[i];
        }
    }

    for i in 0..3 {
        for j in 0..5 {
            let expected: i32 = if (a[i] - a_min) == j { 1 } else { 0 };
            assert_eq!(result[i][j], expected);
        }
    }

    // env::commit(&output);
}
