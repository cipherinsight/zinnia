// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read a (3x5)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..5 {
            row.push(env::read::<i32>());
        }
        a.push(row);
    }

    // read result (scalar)
    let result: i32 = env::read::<i32>();

    // comparison = (a == a[0]) elementwise, then per-row all()
    for r in 0..3 {
        let mut all_equal_row: bool = true;
        for c in 0..5 {
            let eq = a[r as usize][c as usize] == a[0][c as usize];
            all_equal_row = all_equal_row && eq;
        }
        // assert np.all(comparison, axis=-1)[r] == (result == 1)
        assert_eq!(all_equal_row, result == 1);
    }

    // env::commit(&output);
}
