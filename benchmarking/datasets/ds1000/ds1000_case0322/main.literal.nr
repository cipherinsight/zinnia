// DS-1000 case 0322: literal translation of kernel.py (LITERAL_BASELINES.md).
// Minimum by pairwise scan; argwhere by prefix count and equality-test compaction.
// @outputs: dynamic int[4x2]
fn main(a: [Field; 4]) -> pub [Field; 9] {
    noir_hw::check_inputs(a);
    let mut low = a[0];
    for i in 1..4 { low = noir_hw::min(low, a[i]); }
    let mut mask = [false; 4];
    for i in 0..4 { mask[i] = a[i] == low; }
    let mut count: Field = 0;
    let mut positions = [0; 4];
    for i in 0..4 {
        positions[i] = count;
        count += mask[i] as Field;
    }
    let mut out = [0; 9];
    out[0] = count;
    for j in 0..4 {
        for axis in 0..2 {
            let mut acc: Field = 0;
            for i in 0..4 {
                let keep = mask[i] & (positions[i] == j as Field);
                let coord = if axis == 0 { i / 2 } else { i % 2 };
                acc += keep as Field * coord as Field;
            }
            out[1 + 2 * j + axis] = acc;
        }
    }
    out
}
