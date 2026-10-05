// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let nrows: usize = 5;
    let ncols: usize = 6;

    // read input a (5x6)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..nrows {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..ncols {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        a.push(row);
    }

    // read result (5)
    let mut result: Vec<i32> = Vec::new();
    for _ in 0..nrows {
        result.push(sp1_zkvm::io::read::<i32>());
    }

    // Step 1: fliplr
    let mut flipped: Vec<Vec<i32>> = vec![vec![0; ncols]; nrows];
    for i in 0..nrows {
        for j in 0..ncols {
            flipped[i][j] = a[i][(ncols - 1) - j];
        }
    }

    // Step 2: diagonal extraction
    let mut diag_vals: Vec<i32> = vec![0; nrows];
    for k in 0..nrows {
        diag_vals[k] = flipped[k][k];
    }

    // Step 3: verify
    for k in 0..nrows {
        assert_eq!(result[k], diag_vals[k]);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
