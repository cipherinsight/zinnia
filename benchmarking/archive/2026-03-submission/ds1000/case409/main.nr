fn main(x: [[Field; 3]; 3], y: [[Field; 3]; 3], z: [[Field; 3]; 3]) {
    // z must equal x + y elementwise
    for i in 0..3 {
        for j in 0..3 {
            assert(z[i][j] == x[i][j] + y[i][j], "Mismatch at ({i},{j})");
        }
    }
}
