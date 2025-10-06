fn main(a: [[Field; 5]; 4], result: [[[Field; 2]; 2]; 4]) {
    // Patch size = 2; trim to 4x4 by dropping the last column
    // Block order: row-major over 2x2 blocks, then in-block order (r, c)

    let n_block_rows = 2; // 4 / 2
    let n_block_cols = 2; // (5 -> trimmed to 4) / 2
    let patch = 2;

    for br in 0..n_block_rows {            // block row
        for bc in 0..n_block_cols {        // block col
            let p = br * n_block_cols + bc; // flattened patch index 0..3

            for r in 0..patch {            // within-patch row
                for c in 0..patch {        // within-patch col
                    let src_row = br * patch + r;       // 0..3
                    let src_col = bc * patch + c;       // 0..3 (we never use col = 4)
                    let expected = a[src_row][src_col];
                    assert(
                        result[p][r][c] == expected,
                        "Mismatch at patch {p}, ({r},{c})"
                    );
                }
            }
        }
    }
}
