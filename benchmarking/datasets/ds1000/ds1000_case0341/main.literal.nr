// DS-1000 case 0341, Noir baseline, literal translation of kernel.py
// (LITERAL_BASELINES.md).
// Kernel: `result = []; for value in X.flat: result.append(value)` on
// Integer[5, 6]. X.flat visits the 30 elements in row-major order, which is
// already the array's storage order, so the loop is a straight copy. There
// is no arithmetic to translate.
// @outputs: int[30]

fn main(X: [Field; 30]) -> pub [Field; 30] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(X);
    // for value in X.flat: result.append(value)
    let mut result = [0; 30];
    for i in 0..30 { result[i] = X[i]; }
    result
}
