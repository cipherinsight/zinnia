// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut data: [[f64; 5]; 2] = [[0.0; 5]; 2];
    let mut result: [[f64; 1]; 2] = [[0.0; 1]; 2];

    for i in 0..2 {
        for j in 0..5 {
            data[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    for i in 0..2 {
        for j in 0..1 {
            result[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }

    let bin_size: usize = 3;
    let mut expected: [[f64; 1]; 2] = [[0.0; 1]; 2];

    for i in 0..2 {
        let mut reversed: [f64; 5] = [0.0; 5];
        for j in 0..5 {
            reversed[j] = data[i][4 - j];
        }

        let mut sum = 0.0;
        for j in 0..bin_size {
            sum += reversed[j];
        }

        let mean = sum / (bin_size as f64);
        expected[i][0] = mean;
    }

    for i in 0..2 {
        for j in 0..1 {
            assert!((result[i][j] - expected[i][j]).abs() < 1e-9);
        }
    }
}
