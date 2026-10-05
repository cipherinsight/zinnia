// DS-1000 case 0440, Noir baseline.
// Kernel: X = np.zeros([5, 10]); X[:, i] += np.sqrt(np.diag(Y[i])) for each i
// on Y: Float[10, 5, 5]: X[j, i] = sqrt(Y[i, j, j]) (adding to 0.0 is exact).
// 50 exact floor square roots; a negative diagonal entry has no proof (NumPy
// returns NaN).
// @outputs: float[50]

use noir_hw::sqrt;

fn main(Y: [Field; 250]) -> pub [Field; 50] {
    let mut X: [Field; 50] = [0; 50];
    for j in 0..5 {
        for i in 0..10 {
            X[j * 10 + i] = sqrt(Y[i * 25 + j * 6]);
        }
    }
    X
}
