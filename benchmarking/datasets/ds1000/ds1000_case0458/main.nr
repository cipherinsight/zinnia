// DS-1000 case 0458, optimized Noir baseline.
// dim = 6 is the input-width constant; the source body never reads it.
// Compute only the strict upper triangle of the norm matrix. One sum-of-squares rescale, then sqrt.
// The rescaled squared norm is in the Q32 window in both variants.
// Lower-triangle and diagonal zeros are compile-time outputs, not public instances.
// @outputs: const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; float[1]; float[1]; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; float[1]; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; float[1]; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0; const float[1] = 0
fn main(a: [Field; 48]) -> pub [Field; 28] {
    noir_hw::check_inputs(a);
    let mut distance = [0; 64];
    for i in 0..8 {
        for j in i + 1..8 {
            let mut sum: Field = 0;
            for k in 0..6 {
                let d = a[j * 6 + k] - a[i * 6 + k];
                sum += d * d;
            }
            let squared = noir_hw::rescale(sum);
            let norm = noir_hw::sqrt(squared);
            distance[i * 8 + j] = norm;
        }
    }
    let mut result = [0; 28];
    let mut at = 0;
    for i in 0..8 {
        for j in 0..8 {
            if i < j {
                result[at] = distance[i * 8 + j];
                at += 1;
            }
        }
    }
    result
}
