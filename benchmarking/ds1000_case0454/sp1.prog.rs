// DS-1000 case 454 — SP1 guest.
// Kernel: `X / np.abs(X).max(axis=1).reshape(-1, 1)` on Integer[5, 4].
// Linf row-normalisation.

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
    let mut abs_f: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            let v = x[i][j] as f64;
            x_f[i][j] = v;
            abs_f[i][j] = v.abs();
        }
    }

    let mut result: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        let mut linf: f64 = abs_f[i][0];
        for j in 1..4 {
            if abs_f[i][j] > linf {
                linf = abs_f[i][j];
            }
        }
        assert!(linf != 0.0);
        for j in 0..4 {
            result[i][j] = x_f[i][j] / linf;
        }
    }

    for i in 0..5 {
        for j in 0..4 {
            assert!((result[i][j] - claimed[i][j]).abs() < 0.001);
        }
    }
}
