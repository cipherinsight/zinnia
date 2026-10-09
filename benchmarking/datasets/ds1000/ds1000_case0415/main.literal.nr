// DS-1000 case 0415: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reshape is wiring; max uses an index-order pairwise scan with fresh bounds.
// @outputs: float[2]
fn main(data: [Field; 8]) -> pub [Field; 2] {
    noir_hw::check_inputs(data);
    let mut result = [0; 2];
    for b in 0..2 {
        let mut best = data[b * 4];
        for k in 1..4 { best = noir_hw::max(best, data[b * 4 + k]); }
        result[b] = best;
    }
    result
}
