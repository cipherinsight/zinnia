// DS-1000 case 0416: literal translation of kernel.py (LITERAL_BASELINES.md).
// Slice and reshape are wiring; each mean is a running sum then constant division.
// @outputs: float[24]
fn main(data: [Field; 48]) -> pub [Field; 24] {
    noir_hw::check_inputs(data);
    let mut result = [0; 24];
    for row in 0..8 {
        for b in 0..3 {
            let mut sum: Field = 0;
            for k in 0..2 { sum += data[row * 6 + b * 2 + k]; }
            result[row * 3 + b] = noir_hw::floor_div_pow2::<1>(sum);
        }
    }
    result
}
