fn main(a: [[Field; 5]; 5], result: [Field; 5]) {
    // Step 1: Horizontally flip each row (fliplr)
    let mut flipped: [[Field; 5]; 5] = [[0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            flipped[i][j] = a[i][4 - j];
        }
    }

    // Step 2: Collect diagonal values from flipped
    let mut diag_vals: [Field; 5] = [0; 5];
    for k in 0..5 {
        diag_vals[k] = flipped[k][k];
    }

    // Step 3: Verify
    for k in 0..5 {
        assert(result[k] == diag_vals[k], "Mismatch at diagonal index {k}");
    }
}
