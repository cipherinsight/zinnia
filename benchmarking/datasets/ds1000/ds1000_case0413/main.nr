// DS-1000 case 0413: optimized baseline.
// Complex values are interleaved (real, imaginary) Q32 components, each checked.
// Exact nonzero imaginary test; length counts complex items, padded unit is two.
// Shared compaction carries both components in stable input order.
// @outputs: dynamic float[10x2]
fn main(x: [Field; 20]) -> pub [Field; 21] {
    noir_hw::check_inputs(x);
    let mut keep: [Field; 10] = [0; 10];
    let mut weights: [[Field; 2]; 10] = [[0; 2]; 10];
    let mut count: Field = 0;
    for i in 0..10 {
        keep[i] = (x[2 * i + 1] != 0) as Field;
        // im is zero exactly when keep is zero, so keep * im is already im.
        weights[i] = [keep[i] * x[2 * i], x[2 * i + 1]];
        count += keep[i];
    }
    let kept = noir_hw::compact_rows(keep, weights);
    let mut result = [0; 21];
    result[0] = count;
    for j in 0..10 {
        result[2 * j + 1] = kept[j][0];
        result[2 * j + 2] = kept[j][1];
    }
    result
}
