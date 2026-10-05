// DS-1000 case 0342, Noir baseline (same kernel as 0340).
// Kernel: `[value for value in X.flat]` on Integer[5, 6]: the 30 elements in
// row-major order.
// @outputs: int[30]

fn main(X: [Field; 30]) -> pub [Field; 30] {
    X
}
