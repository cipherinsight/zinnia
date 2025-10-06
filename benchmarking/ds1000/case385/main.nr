fn main(a: [[Field; 4]; 4], result: [[[Field; 2]; 2]; 4]) {
    // We want 2x2 patches in the order:
    // (rows 0..1, cols 0..1), (rows 2..3, cols 0..1),
    // (rows 0..1, cols 2..3), (rows 2..3, cols 2..3).

    let mut computed: [[[Field; 2]; 2]; 4] = [
        [[0,0],[0,0]],
        [[0,0],[0,0]],
        [[0,0],[0,0]],
        [[0,0],[0,0]],
    ];

    for b in 0..4 {
        let rblock = b & 1;       // 0,1,0,1
        let cblock = b >> 1;      // 0,0,1,1
        for i in 0..2 {
            for j in 0..2 {
                let row = rblock * 2 + i;
                let col = cblock * 2 + j;
                computed[b][i][j] = a[row][col];
            }
        }
    }

    // Verify
    for b in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                assert(result[b][i][j] == computed[b][i][j], "Mismatch at block {b} ({i},{j})");
            }
        }
    }
}
