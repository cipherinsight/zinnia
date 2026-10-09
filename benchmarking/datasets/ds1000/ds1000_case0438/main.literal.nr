// DS-1000 case 0438: literal translation of kernel.py (LITERAL_BASELINES.md).
// corrcoef uses the full sample covariance matrix, dot rescale once, /3, row/column sqrt normalization and clipping.
// @outputs: float[1]
fn main(post: [Field; 4], distance: [Field; 4]) -> pub Field {
    noir_hw::check_inputs(post);
    noir_hw::check_inputs(distance);
    let mut data = [[0; 4]; 2];
    for k in 0..4 { data[0][k] = noir_hw::from_int(post[k]); }
    for k in 0..4 { data[1][k] = noir_hw::from_int(distance[k]); }
    let mut means = [0; 2];
    for i in 0..2 {
        let mut sum: Field = 0;
        for k in 0..4 { sum += data[i][k]; }
        means[i] = noir_hw::floor_div_pow2::<2>(sum);
    }
    let mut centered = [[0; 4]; 2];
    for i in 0..2 { for k in 0..4 { centered[i][k] = data[i][k] - means[i]; } }
    let mut covariance = [[0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut sum: Field = 0;
            for k in 0..4 { sum += centered[i][k] * centered[j][k]; }
            covariance[i][j] = noir_hw::floor_div_const::<3>(noir_hw::rescale(sum));
        }
    }
    let diagonal = [covariance[0][0], covariance[1][1]];
    let mut stddev = [0; 2];
    for i in 0..2 { stddev[i] = noir_hw::sqrt(diagonal[i]); }
    for i in 0..2 { for j in 0..2 { covariance[i][j] = noir_hw::div(covariance[i][j], stddev[i]); } }
    for i in 0..2 { for j in 0..2 { covariance[i][j] = noir_hw::div(covariance[i][j], stddev[j]); } }
    for i in 0..2 {
        for j in 0..2 {
            covariance[i][j] = noir_hw::min(noir_hw::max(covariance[i][j], -4294967296), 4294967296);
        }
    }
    covariance[0][1]
}
