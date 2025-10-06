// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read inputs
    let mut a: Vec<f32> = Vec::new();
    for _ in 0..3 {
        a.push(env::read());
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..3 {
            tmp.push(env::read());
        }
        result.push(tmp);
    }

    // fixed vals = [-0.4, 1.3, 1.5]
    let vals: [f32; 3] = [-0.4_f32, 1.3_f32, 1.5_f32];

    for i in 0..3 {
        for j in 0..3 {
            let expected: i32 = if a[i] == vals[j] { 1 } else { 0 };
            assert_eq!(result[i][j], expected);
        }
    }

    // env::commit(&output);
}
