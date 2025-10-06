// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read data (10)
    let mut data: Vec<f32> = Vec::new();
    for _ in 0..10 {
        data.push(sp1_zkvm::io::read::<f32>());
    }

    // read result (3)
    let mut result: Vec<f32> = Vec::new();
    for _ in 0..3 {
        result.push(sp1_zkvm::io::read::<f32>());
    }

    let bin_size: usize = 3;
    // reverse
    let mut new_data: Vec<f32> = data.clone();
    new_data.reverse();

    // trim to multiple of bin_size
    let n_trim = (10 / bin_size) * bin_size;
    let trimmed: Vec<f32> = new_data[..n_trim].to_vec();

    // reshape (3x3)
    let mut reshaped: Vec<Vec<f32>> = Vec::new();
    for i in 0..3 {
        let mut row: Vec<f32> = Vec::new();
        for j in 0..bin_size {
            row.push(trimmed[i * bin_size + j]);
        }
        reshaped.push(row);
    }

    // mean along axis=1
    let mut bin_data_mean: Vec<f32> = Vec::new();
    for i in 0..3 {
        let mut s: f32 = 0.0;
        for j in 0..bin_size {
            s += reshaped[i][j];
        }
        bin_data_mean.push(s / (bin_size as f32));
    }

    // assert result == expected
    for i in 0..3 {
        assert!((result[i] - bin_data_mean[i]).abs() < 1e-6);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
