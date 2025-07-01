fn main(input: [[Field; 6]; 5], result: [[Field; 4]; 3]) {
    let mut zero_rows: [bool; 5] = [false; 5];
    for i in 0..5 {
        let mut all_zero = true;
        for j in 0..6 {
            let z = input[i][j] == 0;
            all_zero = all_zero & z;
        }
        zero_rows[i] = all_zero;
    }

    let mut zero_cols: [bool; 6] = [false; 6];
    for j in 0..6 {
        let mut all_zero = true;
        for i in 0..5 {
            let z = input[i][j] == 0;
            all_zero = all_zero & z;
        }
        zero_cols[j] = all_zero;
    }

    let flat_result: [Field; 12] = [
        result[0][0], result[0][1], result[0][2], result[0][3],
        result[1][0], result[1][1], result[1][2], result[1][3],
        result[2][0], result[2][1], result[2][2], result[2][3],
    ];

    let mut idx: u128 = 0;
    for i in 0..5 {
        for j in 0..6 {
            if !(zero_rows[i] | zero_cols[j]) {
                let k: u128 = idx as u128;
                assert(input[i][j] == flat_result[k], "Mismatch at flattened index k, position (i,j)");
                idx += 1;
            }
        }
    }

    assert(idx == 12, "Expected 12 non-skipped entries, but found {idx}");
}
