fn main(a: [[[Field; 2]; 3]; 3], b: [[Field;    3]; 3], desired: [[Field; 3]; 3]) {
    for i in 0..3 {
        for j in 0..3 {
            let bij = b[i][j];
            assert((bij == 0) | (bij == 1), "b[i][j] must be 0 or 1");

            if bij == 0 {
                assert(a[i][j][0] == desired[i][j], "Mismatch");
            } else {
                assert(a[i][j][1] == desired[i][j], "Mismatch");
            }
        }
    }
}
