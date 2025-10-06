// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read input arrays
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..3 {
        a.push(sp1_zkvm::io::read::<i32>());
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..5 {
            tmp.push(sp1_zkvm::io::read::<i32>());
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

    // sp1_zkvm::io::commit_slice(&output);
}
