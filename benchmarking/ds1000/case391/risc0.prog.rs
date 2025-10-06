// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read a (3x8)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..8 {
            row.push(env::read::<i32>());
        }
        a.push(row);
    }

    // read result (3x4)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..4 {
            row.push(env::read::<i32>());
        }
        result.push(row);
    }

    let low: usize = 1;
    let high: usize = 5;

    // expected = a[:, low:high]
    for r in 0..3usize {
        for c in 0..(high - low) {
            let src_c = low + c;
            assert_eq!(result[r][c as usize], a[r][src_c as usize]);
        }
    }
}
