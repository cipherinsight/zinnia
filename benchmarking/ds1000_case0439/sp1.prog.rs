// DS-1000 case 439 — SP1 guest.
// Kernel: `X.T[:, :, None] * X.T[:, None]` on Float[10, 20].
// Result: Float[20, 10, 10] where result[k, i, j] = X[i, k] * X[j, k].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut x: [[f64; 20]; 10] = [[0.0; 20]; 10];
    for i in 0..10 {
        for k in 0..20 {
            x[i][k] = sp1_zkvm::io::read::<f64>();
        }
    }
    // Claimed result, flattened in (k, i, j) order, shape [20][10][10].
    let mut claimed: Vec<f64> = Vec::with_capacity(20 * 10 * 10);
    for _ in 0..(20 * 10 * 10) {
        claimed.push(sp1_zkvm::io::read::<f64>());
    }

    // X.T: xt[k][i] = x[i][k]
    let mut xt: [[f64; 10]; 20] = [[0.0; 10]; 20];
    for k in 0..20 {
        for i in 0..10 {
            xt[k][i] = x[i][k];
        }
    }

    let mut result: [[[f64; 10]; 10]; 20] = [[[0.0; 10]; 10]; 20];
    for k in 0..20 {
        for i in 0..10 {
            for j in 0..10 {
                result[k][i][j] = xt[k][i] * xt[k][j];
            }
        }
    }

    let mut idx = 0;
    for k in 0..20 {
        for i in 0..10 {
            for j in 0..10 {
                assert!((result[k][i][j] - claimed[idx]).abs() < 0.001);
                idx += 1;
            }
        }
    }
}
