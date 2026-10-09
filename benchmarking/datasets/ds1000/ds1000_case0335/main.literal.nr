// DS-1000 case 0335: literal translation of kernel.py (LITERAL_BASELINES.md).
// Stack the three arrays, then pairwise max in axis index order with fresh bounds.
// @outputs: float[50]
fn main(a: [Field; 50], b: [Field; 50], c: [Field; 50]) -> pub [Field; 50] {
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    noir_hw::check_inputs(c);
    let arrays = [a, b, c];
    let mut result = [0; 50];
    for i in 0..50 {
        let mut best = arrays[0][i];
        for j in 1..3 { best = noir_hw::max(best, arrays[j][i]); }
        result[i] = best;
    }
    result
}
