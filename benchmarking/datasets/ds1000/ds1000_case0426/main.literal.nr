// DS-1000 case 0426: literal translation of kernel.py (LITERAL_BASELINES.md).
// Bitwise AND is independently decomposed per broadcast element; materialize masked integers, then compare and cast.
// @outputs: int[300]
global M: u32 = 15;
fn main(a: [Field; 20]) -> pub [Field; 300] {
    noir_hw::check_inputs(a);
    let mut masked = [0; 300];
    for i in 0..20 {
        for j in 0..M {
            let bits: [bool; 64] = (a[i] + noir_hw::OFFSET).to_le_bits();
            masked[i * M + j] = (bits[M - 1 - j] as Field) * ((1 as u64) << ((M - 1 - j) as u64)) as Field;
        }
    }
    let mut positive = [0; 300];
    for i in 0..300 { positive[i] = noir_hw::gt(masked[i], 0) as Field; }
    positive
}
