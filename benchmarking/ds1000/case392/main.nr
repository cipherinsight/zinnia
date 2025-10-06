fn main(a: [[Field; 8]; 3], result: [[Field; 8]; 2]) {
    // Extract rows in [low, high) = [0, 2)
    let low = 0;
    let high = 2;

    for i in low..high {
        let out_row = i - low; // 0..1
        for j in 0..8 {
            assert(
                result[out_row][j] == a[i][j],
                "Mismatch at row {i}, col {j}"
            );
        }
    }
}
