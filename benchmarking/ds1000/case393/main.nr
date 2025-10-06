fn main(a: [[Field; 8]; 3], result: [[Field; 7]; 3]) {
    // Extract columns in [low, clamped_high) with low=1, clamped_high=8
    let low = 1;
    let clamped_high = 8;

    for i in 0..3 {
        for j in low..clamped_high {
            let out_col = j - low; // 0..6
            assert(
                result[i][out_col] == a[i][j],
                "Mismatch at row {i}, col {j}"
            );
        }
    }
}
