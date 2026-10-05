// DS-1000 case 0343, Noir baseline.
// Kernel: `[value for value in X.T.flat]` on Integer[5, 6]: the 30 elements in
// column-major order, X[i, j] at position j * 5 + i.
// @outputs: int[30]

fn main(X: [Field; 30]) -> pub [Field; 30] {
    let mut transposed = [0; 30];
    for j in 0..6 {
        for i in 0..5 {
            transposed[j * 5 + i] = X[i * 6 + j];
        }
    }
    transposed
}
