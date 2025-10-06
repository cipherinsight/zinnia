// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read input arrays
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..3 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        a.push(tmp);
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..6 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..5 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        result.push(tmp);
    }

    let a_min: i32 = 0;

    // flatten a in C order
    let flat: [i32; 6] = [
        a[0][0], a[0][1], a[0][2],
        a[1][0], a[1][1], a[1][2]
    ];

    for i in 0..6 {
        for j in 0..5 {
            let expected: i32 = if (flat[i] - a_min) == j { 1 } else { 0 };
            assert_eq!(result[i][j], expected);
        }
    }

    // sp1_zkvm::io::commit_slice(&output);
}
