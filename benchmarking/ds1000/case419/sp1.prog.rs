// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read data (2x5)
    let mut data: Vec<Vec<f32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<f32> = Vec::new();
        for _ in 0..5 {
            row.push(sp1_zkvm::io::read::<f32>());
        }
        data.push(row);
    }

    // read result (2x1)
    let mut result: Vec<Vec<f32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<f32> = Vec::new();
        for _ in 0..1 {
            row.push(sp1_zkvm::io::read::<f32>());
        }
        result.push(row);
    }

    let bin_size: usize = 3;
    let ncol: usize = (5 / bin_size) * bin_size;

    // Reverse each row
    let mut new_data: Vec<Vec<f32>> = Vec::new();
    for i in 0..2 {
        let mut row = data[i].clone();
        row.reverse();
        new_data.push(row);
    }

    // Trim columns to multiple of bin_size
    let mut trimmed: Vec<Vec<f32>> = Vec::new();
    for i in 0..2 {
        let mut row: Vec<f32> = Vec::new();
        for j in 0..ncol {
            row.push(new_data[i][j]);
        }
        trimmed.push(row);
    }

    // Compute mean along last axis (reshape (2,1,3))
    let mut bin_data_mean: Vec<Vec<f32>> = Vec::new();
    for i in 0..2 {
        let mut row: Vec<f32> = Vec::new();
        let mut s: f32 = 0.0;
        for j in 0..bin_size {
            s += trimmed[i][j];
        }
        row.push(s / (bin_size as f32));
        bin_data_mean.push(row);
    }

    // Reverse result (shape (2,1))
    let mut reversed_result: Vec<Vec<f32>> = bin_data_mean.clone();
    for i in 0..2 {
        reversed_result[i].reverse();
    }

    // Assert equality
    for i in 0..2 {
        for j in 0..1 {
            assert!((result[i][j] - reversed_result[i][j]).abs() < 1e-6);
        }
    }

    // sp1_zkvm::io::commit_slice(&output);
}
