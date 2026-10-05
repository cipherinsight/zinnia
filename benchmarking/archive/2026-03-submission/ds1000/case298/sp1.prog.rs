// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read inputs
    let mut a: Vec<f32> = Vec::new();
    for _ in 0..3 {
        a.push(sp1_zkvm::io::read::<f32>());
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..3 {
            tmp.push(sp1_zkvm::io::read::<i32>());
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

    // sp1_zkvm::io::commit_slice(&output);
}
