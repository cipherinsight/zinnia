// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    let nrows: usize = 5;
    let ncols: usize = 6;

    // read input a (5x6)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..nrows {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..ncols {
            row.push(env::read());
        }
        a.push(row);
    }

    // read result (2x5)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..5 {
            row.push(env::read());
        }
        result.push(row);
    }

    // dim = min(nrows, ncols)
    let dim: usize = if nrows < ncols { nrows } else { ncols };

    // b = a[:dim, :dim]
    let mut b: Vec<Vec<i32>> = vec![vec![0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            b[i][j] = a[i][j];
        }
    }

    // main diagonal
    let mut main_diag: Vec<i32> = vec![0; dim];
    for i in 0..dim {
        main_diag[i] = b[i][i];
    }

    // flipped = fliplr(b)
    let mut flipped: Vec<Vec<i32>> = vec![vec![0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            flipped[i][j] = b[i][(dim - 1) - j];
        }
    }

    // anti diagonal = diag(fliplr(b))
    let mut anti_diag: Vec<i32> = vec![0; dim];
    for i in 0..dim {
        anti_diag[i] = flipped[i][i];
    }

    // stacked = vstack(main_diag, anti_diag) -> (2, dim)
    let mut stacked: Vec<Vec<i32>> = vec![vec![0; dim], vec![0; dim]];
    for j in 0..dim {
        stacked[0][j] = main_diag[j];
        stacked[1][j] = anti_diag[j];
    }

    // compare with result
    for r in 0..2 {
        for c in 0..dim {
            assert_eq!(result[r][c], stacked[r][c]);
        }
    }

    // env::commit(&output);
}
