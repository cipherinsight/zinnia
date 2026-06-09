// DS-1000 case 449 — SP1 guest.
// Kernel: `dists = np.vstack(([x_dists.T], [y_dists.T])).T` on Float[3,4] × Float[3,4].
// Result: Float[3,4,2] where dists[i][j][0] = x_dists[i][j], dists[i][j][1] = y_dists[i][j].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut x_dists: [[f64; 4]; 3] = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            x_dists[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    let mut y_dists: [[f64; 4]; 3] = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            y_dists[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    let mut claimed: [[[f64; 2]; 4]; 3] = [[[0.0; 2]; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..2 {
                claimed[i][j][k] = sp1_zkvm::io::read::<f64>();
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
