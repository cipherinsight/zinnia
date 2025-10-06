// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [i32; 3] = [0; 3];
    let mut result: [[i32; 4]; 3] = [[0; 4]; 3];

    for i in 0..3 {
        a[i] = sp1_zkvm::io::read::<i32>();
    }
    for i in 0..3 {
        for j in 0..4 {
            result[i][j] = sp1_zkvm::io::read::<i32>();
        }
    }

    for i in 0..3 {
        for j in 0..4 {
            let expected = if a[i] == j as i32 { 1 } else { 0 };
            assert_eq!(result[i][j], expected);
        }
    }
}
