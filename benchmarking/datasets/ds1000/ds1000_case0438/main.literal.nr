// DS-1000 case 0438: literal translation of kernel.py (LITERAL_BASELINES.md).
// corrcoef forms the full matrix, every product and sum in index order.
// Pearson uses optimized exact integer centred moments, one root and quotient
// per entry. Diagonal perfect-square roots use integer units to avoid adding
// a Q32 root-window restriction absent from the optimized statement.
// @outputs: float[1]
fn main(post: [Field; 4], distance: [Field; 4]) -> pub Field {
    noir_hw::check_inputs(post);
    noir_hw::check_inputs(distance);
    let data = [post, distance];
    let mut sums = [0; 2];
    for i in 0..2 {
        for k in 0..4 { sums[i] += data[i][k]; }
    }
    let mut moments = [[0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut products = [0; 4];
            for k in 0..4 { products[k] = data[i][k] * data[j][k]; }
            let mut sum: Field = 0;
            for k in 0..4 { sum += products[k]; }
            moments[i][j] = 4 * sum - sums[i] * sums[j];
        }
    }
    let diagonal = [moments[0][0], moments[1][1]];
    for i in 0..2 { diagonal[i].assert_max_bit_size::<64>(); }
    let mut correlation = [[0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let product = diagonal[i] * diagonal[j];
            if i == j {
                let root = noir_hw::isqrt(product);
                correlation[i][j] = noir_hw::divmod_pos(moments[i][j] * noir_hw::SCALE, root).0;
            } else {
                let root = noir_hw::sqrt(noir_hw::from_int(product));
                correlation[i][j] = noir_hw::divmod_pos(noir_hw::from_int(moments[i][j]) * noir_hw::SCALE, root).0;
            }
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            correlation[i][j] = noir_hw::min(noir_hw::max(correlation[i][j], -4294967296), 4294967296);
        }
    }
    correlation[0][1]
}
