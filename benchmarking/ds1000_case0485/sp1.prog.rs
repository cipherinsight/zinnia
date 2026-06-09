// DS-1000 case 485 — SP1 guest program.
// kernel.py: result = np.tensordot(A, B, axes=((2),(0))) for A: Float[5,6,3], B: Float[3,3].
// Output shape: (5, 6, 3). result[i, j, k] = sum_{p=0..3} A[i,j,p] * B[p,k].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // 5*6*3 = 90 entries of A.
    let mut a: [[[f64; 3]; 6]; 5] = [[[0.0; 3]; 6]; 5];
    for i in 0..5 {
        for j in 0..6 {
            for p in 0..3 {
                a[i][j][p] = sp1_zkvm::io::read::<f64>();
            }
        }
    }
    let mut b: [[f64; 3]; 3] = [[0.0; 3]; 3];
    for p in 0..3 {
        for k in 0..3 {
            b[p][k] = sp1_zkvm::io::read::<f64>();
        }
    }
    let mut claimed: [f64; 90] = [0.0; 90];
    for idx in 0..90 {
        claimed[idx] = sp1_zkvm::io::read::<f64>();
    }

    for i in 0..5 {
        for j in 0..6 {
            for k in 0..3 {
                let mut acc: f64 = 0.0;
                for p in 0..3 {
                    acc += a[i][j][p] * b[p][k];
                }
                let idx = i * 18 + j * 3 + k;
                assert!((acc - claimed[idx]).abs() <= 0.005);
            }
        }
    }
}
