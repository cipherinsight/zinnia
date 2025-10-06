// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read a (3x8)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..8 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        a.push(row);
    }

    // read result (2x8)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..8 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        result.push(row);
    }

    let low: usize = 0;
    let high: usize = 2;

    // expected = a[low:high, :]
    for r in 0..(high - low) {
        for c in 0..8usize {
            let src_r = low + r;
            assert_eq!(result[r][c], a[src_r][c]);
        }
    }
}
