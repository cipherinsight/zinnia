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

    // comparison = (a == a[0]) elementwise, then np.all over all elements
    let mut computed: bool = true;
    for r in 0..3 {
        for c in 0..5 {
            let eq = a[r as usize][c as usize] == a[0][c as usize];
            computed = computed && eq;
        }
    }

    // assert (result == 1) == computed
    assert_eq!(result == 1, computed);

    // env::commit(&output);
}
