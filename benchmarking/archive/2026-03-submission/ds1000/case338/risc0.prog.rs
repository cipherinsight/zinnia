// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    let n: usize = 5;

    // read input a (5x5)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..n {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..n {
            row.push(env::read());
        }
        a.push(row);
    }

    // read result (2x5)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..n {
            row.push(env::read());
        }
        result.push(row);
    }

    // main diagonal
    let mut main_diag: Vec<i32> = vec![0; n];
    for i in 0..n {
        main_diag[i] = a[i][i];
    }

    // fliplr
    let mut flipped: Vec<Vec<i32>> = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            flipped[i][j] = a[i][(n - 1) - j];
        }
    }

    // anti diagonal = diag(fliplr(a))
    let mut anti_diag: Vec<i32> = vec![0; n];
    for i in 0..n {
        anti_diag[i] = flipped[i][i];
    }

    // stacked = vstack(main_diag, anti_diag) -> (2, n)
    let mut stacked: Vec<Vec<i32>> = vec![vec![0; n], vec![0; n]];
    for j in 0..n {
        stacked[0][j] = main_diag[j];
        stacked[1][j] = anti_diag[j];
    }

    // compare with result
    for r in 0..2 {
        for c in 0..n {
            assert_eq!(result[r][c], stacked[r][c]);
        }
    }

    // env::commit(&output);
}
