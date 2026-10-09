// DS-1000 case 0419: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reverse the six columns, reshape into pairs, and compute means in row order.
// Slice/reversal/reshape are wiring; each mean sums then floor-divides by two.
// Reverse each row of three means at the end.
// @outputs: float[24]
fn main(data: [Field; 48]) -> pub [Field; 24] {
    noir_hw::check_inputs(data);
    // new_data = data[:, ::-1]
    let mut new_data = [0; 48];
    for row in 0..8 {
        for col in 0..6 { new_data[row * 6 + col] = data[row * 6 + 5 - col]; }
    }
    let mut result = [0; 24];
    for row in 0..8 {
        for b in 0..3 {
            let mut sum: Field = 0;
            for k in 0..2 { sum += new_data[row * 6 + b * 2 + k]; }
            result[row * 3 + b] = noir_hw::floor_div_pow2::<1>(sum);
        }
    }
    // Final [:, ::-1].
    let mut reordered = [0; 24];
    for row in 0..8 {
        for b in 0..3 { reordered[row * 3 + b] = result[row * 3 + 2 - b]; }
    }
    reordered
}
