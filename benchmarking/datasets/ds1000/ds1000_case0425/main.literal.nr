// DS-1000 case 0425: literal translation of kernel.py (LITERAL_BASELINES.md).
// Bitwise AND is independently decomposed per broadcast element; materialize masked integers, then compare and cast.
// @outputs: int[220]
global M: u32 = 11;
fn main(a: [Field; 20]) -> pub [Field; 220] {
    noir_hw::check_inputs(a);
    let mut masked = [0; 220];
    for i in 0..20 {
        for j in 0..M {
            let bits: [bool; 64] = (a[i] + noir_hw::OFFSET).to_le_bits();
            masked[i * M + j] = (bits[M - 1 - j] as Field) * ((1 as u64) << ((M - 1 - j) as u64)) as Field;
        }
    }
    let mut positive = [0; 220];
    for i in 0..220 { positive[i] = noir_hw::gt(masked[i], 0) as Field; }
    positive
}
