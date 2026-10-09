// DS-1000 case 0477, literal translation of kernel.py (LITERAL_BASELINES.md).
// Construct the full diagonal matrix; omit constant outputs from instances (L7).
// @outputs: float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]
fn main(i: [Field; 4]) -> pub [Field; 4] {
    noir_hw::check_inputs(i);
    // i = np.diag(i)
    let mut diagonal = [0; 16];
    for k in 0..4 { diagonal[k * 4 + k] = i[k]; }
    let mut result = [0; 4];
    for k in 0..4 { result[k] = diagonal[k * 4 + k]; }
    result
}
