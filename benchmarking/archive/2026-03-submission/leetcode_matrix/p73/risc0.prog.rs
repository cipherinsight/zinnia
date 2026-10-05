use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut matrix: Vec<Vec<i32>> = Vec::new();
    for i in 0..8 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..10 {
            tmp.push(env::read());
        }
        matrix.push(tmp);
    }
    let mut sol: Vec<Vec<i32>> = Vec::new();
    for i in 0..8 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..10 {
            tmp.push(env::read());
        }
        sol.push(tmp);
    }

    let m = 8;
    let n = 10;

    for i in 0..m {
        for j in 0..n {
            if matrix[i][j] == 0 {
                // Ensure the entire column j in sol is 0
                for row in 0..m {
                    assert_eq!(
                        sol[row][j], 0,
                        "Expected sol[{}][{}] to be 0, but got {}",
                        row, j, sol[row][j]
                    );
                }
                // Ensure the entire row i in sol is 0
                for col in 0..n {
                    assert_eq!(
                        sol[i][col], 0,
                        "Expected sol[{}][{}] to be 0, but got {}",
                        i, col, sol[i][col]
                    );
                }
            }
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
