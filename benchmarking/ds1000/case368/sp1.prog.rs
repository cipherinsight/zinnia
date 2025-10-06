// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read a (3x5)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..5 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        a.push(row);
    }

    // read result (scalar)
    let result: i32 = sp1_zkvm::io::read::<i32>();

    // comparison = (a == a[0]) elementwise, then per-row all()
    for r in 0..3 {
        let mut all_equal_row: bool = true;
        for c in 0..5 {
            let eq = a[r as usize][c as usize] == a[0][c as usize];
            all_equal_row = all_equal_row && eq;
        }
        assert_eq!(all_equal_row, result == 1);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
