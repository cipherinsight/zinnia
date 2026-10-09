// DS-1000 case 0328: literal translation of kernel.py (LITERAL_BASELINES.md).
// Shape condition is constant. Full copy and slice assignment are wire aliases; only retained inputs need domain checks (L7).
// @outputs: float[3]; const float[2] = [0.0, 0.0]
fn main(A: [Field; 5]) -> pub [Field; 3] {
    let length: u32 = 3;
    // length > A.shape[0] is false at compile time.
    let mut result = A;
    for i in length..5 { result[i] = 0; }
    for i in 0..3 { noir_hw::check_input(A[i]); }
    [result[0], result[1], result[2]]
}
