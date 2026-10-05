// DS-1000 case 453 — SP1 guest.
// Kernel: `X / np.sqrt((X*X).sum(axis=-1)).reshape(-1,1)` on Integer[5,4].
// L2 row-normalisation.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut x: [[i64; 4]; 5] = [[0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            x[i][j] = sp1_zkvm::io::read::<i64>();
        }
    }
    let mut claimed: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            claimed[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }

    let mut x_f: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            x_f[i][j] = x[i][j] as f64;
        }
    }

    let mut result: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        let mut ss: f64 = 0.0;
        for j in 0..4 {
            ss = ss + x_f[i][j] * x_f[i][j];
        }
        let l2 = ss.sqrt();
        assert!(l2 != 0.0);
        for j in 0..4 {
            result[i][j] = x_f[i][j] / l2;
        }
    }

    for i in 0..5 {
        for j in 0..4 {
            assert!((result[i][j] - claimed[i][j]).abs() < 0.001);
        }
    }
}
