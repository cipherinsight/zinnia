// DS-1000 case 0457, literal translation of kernel.py (LITERAL_BASELINES.md).
// dim = 6 is the input-width constant; the source body never reads it.
// Materialize all differences, then every norm; no symmetry or skipped diagonal.
// Running sum of squared terms, one rescale, then sqrt, as optimized arithmetic.
// Diagonal zeros are omitted from public instances only (L7).
// @outputs: const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0
fn main(a: [Field; 48]) -> pub [Field; 56] {
    noir_hw::check_inputs(a);
    // a - a[:, None], broadcast to (8, 8, 6).
    let mut difference = [0; 384];
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..6 {
                difference[(i * 8 + j) * 6 + k] = a[j * 6 + k] - a[i * 6 + k];
            }
        }
    }
    // np.linalg.norm(..., axis=-1).
    let mut norms = [0; 64];
    for i in 0..64 {
        let mut sum: Field = 0;
        for k in 0..6 {
            let d = difference[i * 6 + k];
            let term = d * d;
            sum += term;
        }
        let squared = noir_hw::rescale(sum);
        norms[i] = noir_hw::sqrt(squared);
    }
    let mut result = [0; 56];
    let mut at = 0;
    for i in 0..8 {
        for j in 0..8 {
            if i != j {
                result[at] = norms[i * 8 + j];
                at += 1;
            }
        }
    }
    result
}
