fn main(a: [[Field; 6]; 5], result: [Field; 5]) {
    let nrows = 5;
    let ncols = 6;

    // Step 1: flip each row horizontally (fliplr)
    let mut flipped: [[Field; 6]; 5] = [[0; 6]; 5];
    for i in 0..nrows {
        for j in 0..ncols {
            flipped[i][j] = a[i][(ncols - 1) - j];
        }
    }

    // Step 2: collect diagonal values from flipped
    let mut diag_vals: [Field; 5] = [0; 5];
    for k in 0..nrows {
        diag_vals[k] = flipped[k][k];
    }

    // Step 3: verify equality
    for k in 0..nrows {
        assert(result[k] == diag_vals[k], "Mismatch at diag index {k}");
    }
}
