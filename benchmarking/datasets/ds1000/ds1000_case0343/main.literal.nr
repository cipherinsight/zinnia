// DS-1000 case 0343, literal translation of kernel.py (LITERAL_BASELINES.md).
// X.T.flat visits columns first; append each input wire in that order.
// @outputs: int[30]

fn main(X: [Field; 30]) -> pub [Field; 30] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(X);
    // for value in X.T.flat: result.append(value)
    let mut result = [0; 30];
    for j in 0..6 {
        for i in 0..5 { result[j * 5 + i] = X[i * 6 + j]; }
    }
    result
}
