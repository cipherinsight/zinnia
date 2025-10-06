fn main(a: [[[Field; 3]; 2]; 4], result: [[Field; 6]; 4]) {
    // a has shape (n=4, nrows=2, ncols=3)
    // Tile the 4 (2x3) blocks into a (4x6) array in row-major block order.
    // Block grid is 2x2: blocks 0,1 on the top row; blocks 2,3 on the bottom row.

    let nrows = 2;
    let ncols = 3;

    let n_block_rows = 2; // 4 / 2
    let n_block_cols = 2; // 6 / 3

    for br in 0..n_block_rows {          // block row
        for bc in 0..n_block_cols {      // block col
            let p = br * n_block_cols + bc; // block index 0..3

            for r in 0..nrows {          // within-block row (0..1)
                for c in 0..ncols {      // within-block col (0..2)
                    let R = br * nrows + r;   // output row 0..3
                    let C = bc * ncols + c;   // output col 0..5
                    assert(
                        result[R][C] == a[p][r][c],
                        "Mismatch at out({R},{C}) from block {p}, in({r},{c})"
                    );
                }
            }
        }
    }
}
