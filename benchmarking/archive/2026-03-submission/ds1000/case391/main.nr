fn main(a: [[Field; 8]; 3], result: [[Field; 4]; 3]) {
    // Extract columns in [low, high) = [1, 5)
    let low = 1;
    let high = 5;

    for i in 0..3 {
        for j in low..high {
            let out_col = j - low; // 0..3
            assert(
                result[i][out_col] == a[i][j],
                "Mismatch at row {i}, col {j}"
            );
        }
    }
}
