fn main(a: [[Field; 5]; 5], result: [[Field; 5]; 2]) {
    let n = 5;

    // Step 1: main diagonal
    let mut main_diag: [Field; 5] = [0; 5];
    for i in 0..n {
        main_diag[i] = a[i][i];
    }

    // Step 2: anti-diagonal (fliplr + diag)
    let mut anti_diag: [Field; 5] = [0; 5];
    for i in 0..n {
        anti_diag[i] = a[i][(n - 1) - i];
    }

    // Step 3: verify both diagonals stacked
    for j in 0..n {
        assert(result[0][j] == main_diag[j], "Main diagonal mismatch at index {j}");
        assert(result[1][j] == anti_diag[j], "Anti-diagonal mismatch at index {j}");
    }
}
