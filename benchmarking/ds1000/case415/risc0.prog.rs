// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read data
    let mut data: Vec<i32> = Vec::new();
    for _ in 0..10 {
        data.push(env::read());
    }

    // read result
    let mut result: Vec<i32> = Vec::new();
    for _ in 0..3 {
        result.push(env::read());
    }

    let bin_size: usize = 3;
    let n_bins: usize = (10 / bin_size) * bin_size;
    let trimmed: Vec<i32> = data[..n_bins].to_vec();

    // reshape to (3, 3)
    let mut reshaped: Vec<Vec<i32>> = Vec::new();
    for i in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for j in 0..bin_size {
            row.push(trimmed[i * bin_size + j]);
        }
        reshaped.push(row);
    }

    // max along axis=1
    let mut bin_data_max: Vec<i32> = Vec::new();
    for i in 0..3 {
        let mut m: i32 = reshaped[i][0];
        for j in 1..bin_size {
            if reshaped[i][j] > m {
                m = reshaped[i][j];
            }
        }
        bin_data_max.push(m);
    }

    // assert result == expected
    for i in 0..3 {
        assert_eq!(result[i as usize], bin_data_max[i as usize]);
    }

    // env::commit(&output);
}
