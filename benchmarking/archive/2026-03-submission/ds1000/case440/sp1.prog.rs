// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // Read Y[4][3][3]
    let mut y: [[[f32; 3]; 3]; 4] = [[[0.0; 3]; 3]; 4];
    for i in 0..4 {
        for j in 0..3 {
            for k in 0..3 {
                y[i][j][k] = sp1_zkvm::io::read::<f32>();
            }
        }
    }

    // Read X[3][4]
    let mut x: [[f32; 4]; 3] = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            x[i][j] = sp1_zkvm::io::read::<f32>();
        }
    }

    // Verify diag(Y[i]) = X[:, i]^2
    for i in 0..4 {
        for j in 0..3 {
            let expected_squared = y[i][j][j];
            assert!((x[j][i] * x[j][i] - expected_squared).abs() < 1e-6);
        }
    }
}
