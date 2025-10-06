fn main(a: [[Field; 6]; 5], result: [[Field; 5]; 2]) {
    // dim = min(5, 6) = 5
    let dim = 5;

    // Main diagonal from the top-left 5x5 block: a[i][i]
    let mut main_diag: [Field; 5] = [0; 5];
    for i in 0..dim {
        main_diag[i] = a[i][i];
    }

    // Anti-diagonal via diag(fliplr(b)) == b[i][dim-1-i] == a[i][dim-1-i]
    let mut anti_diag: [Field; 5] = [0; 5];
    for i in 0..dim {
        anti_diag[i] = a[i][(dim - 1) - i];
    }

    // Verify stacked diagonals
    for j in 0..dim {
        assert(result[0][j] == main_diag[j], "Main diag mismatch at {j}");
        assert(result[1][j] == anti_diag[j], "Anti diag mismatch at {j}");
    }
}
