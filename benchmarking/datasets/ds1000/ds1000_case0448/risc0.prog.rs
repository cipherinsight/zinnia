// DS-1000 case 448 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut x_dists: [[f64; 4]; 3] = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            x_dists[i][j] = env::read();
        }
    }
    let mut y_dists: [[f64; 4]; 3] = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            y_dists[i][j] = env::read();
        }
    }
    let mut claimed: [[[f64; 2]; 4]; 3] = [[[0.0; 2]; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..2 {
                claimed[i][j][k] = env::read();
            }
        }
    }

    let mut dists: [[[f64; 2]; 4]; 3] = [[[0.0; 2]; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            dists[i][j][0] = x_dists[i][j];
            dists[i][j][1] = y_dists[i][j];
        }
    }

    for i in 0..3 {
        for j in 0..4 {
            for k in 0..2 {
                assert!((dists[i][j][k] - claimed[i][j][k]).abs() < 0.001);
            }
        }
    }
}
