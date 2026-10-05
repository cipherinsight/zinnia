// DS-1000 case 0440 — SP1 guest program.
// Kernel: X = zeros((M, N)); for i in 0..N: X[:, i] += sqrt(diag(Y[i])).
// Y Float[N, M, K]; assumes M==K for diag-vs-column alignment. Concrete N=2, M=K=3.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 3;
    let mut y = [[[0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K { y[i][j][k] = sp1_zkvm::io::read::<f64>(); } } }
    let mut x = [[0.0f64; N]; M];
    for i in 0..N {
        for j in 0..M {
            x[j][i] += y[i][j][j].sqrt();
        }
    }
    for j in 0..M {
        for i in 0..N {
            let claim: f64 = sp1_zkvm::io::read::<f64>();
            assert!((x[j][i] - claim).abs() < 1e-9);
        }
    }
}
