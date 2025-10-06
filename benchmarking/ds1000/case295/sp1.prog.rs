// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..3 {
        a.push(sp1_zkvm::io::read::<i32>());
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..4 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        result.push(tmp);
    }

    for i in 0..3 {
        for j in 0..4 {
            let expected: i32 = if a[i] == j { 1 } else { 0 };
            assert_eq!(result[i as usize][j as usize], expected);
        }
    }

    // sp1_zkvm::io::commit_slice(&output);
}
