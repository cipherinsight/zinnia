// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut data: [f64; 10] = [0.0; 10];
    let mut result: [f64; 3] = [0.0; 3];

    for i in 0..10 {
        data[i] = sp1_zkvm::io::read::<f64>();
    }
    for i in 0..3 {
        result[i] = sp1_zkvm::io::read::<f64>();
    }

    let bin_size: usize = 3;
    let mut reversed: [f64; 10] = [0.0; 10];
    for i in 0..10 {
        reversed[i] = data[9 - i];
    }

    let mut expected: [f64; 3] = [0.0; 3];
    for i in 0..3 {
        let start = i * bin_size;
        let mut sum = 0.0;
        for j in 0..bin_size {
            sum += reversed[start + j];
        }
        expected[i] = sum / (bin_size as f64);
    }

    for i in 0..3 {
        assert!((result[i] - expected[i]).abs() < 1e-9);
    }
}
