// pythran_euclidean_distance_square__euclidean_distance_square: literal translation of kernel.py (LITERAL_BASELINES.md).
// Only np.dot fuses rescaling; square arrays round individually, additions remain left-associated.

// @outputs: float[R1 * R2]

use noir_hw::{check_inputs, dot};

// Sizes (hw_check.sizes): from fixture.json.
global R1: u32 = 1; // = x1_0
global R2: u32 = 5; // = x2_0
global D: u32 = 3; // = x1_1

fn main(x1: [[Field; D]; R1], x2: [[Field; D]; R2]) -> pub [[Field; R2]; R1] {
    for i in 0..R1 { check_inputs(x1[i]); } for j in 0..R2 { check_inputs(x2[j]); }
    let mut gram = [[0; R2]; R1];
    for i in 0..R1 { for j in 0..R2 { gram[i][j] = dot(x1[i], x2[j]); } }
    let mut scaled = [[0; R2]; R1];
    for i in 0..R1 { for j in 0..R2 { scaled[i][j] = -2*gram[i][j]; } }
    let mut squares1 = [[0; D]; R1];
    for i in 0..R1 { for k in 0..D { squares1[i][k] = noir_hw::mul(x1[i][k], x1[i][k]); } }
    let mut sums1 = [0; R1];
    for i in 0..R1 { for k in 0..D { sums1[i] += squares1[i][k]; } }
    let mut first = [[0; R2]; R1];
    for i in 0..R1 { for j in 0..R2 { first[i][j] = scaled[i][j]+sums1[i]; } }
    let mut squares2 = [[0; D]; R2];
    for j in 0..R2 { for k in 0..D { squares2[j][k] = noir_hw::mul(x2[j][k], x2[j][k]); } }
    let mut sums2 = [0; R2];
    for j in 0..R2 { for k in 0..D { sums2[j] += squares2[j][k]; } }
    let mut result = [[0; R2]; R1];
    for i in 0..R1 { for j in 0..R2 { result[i][j] = first[i][j]+sums2[j]; } }
    result
}
