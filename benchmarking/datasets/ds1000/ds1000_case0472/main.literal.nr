// DS-1000 case 0472: literal translation of kernel.py (LITERAL_BASELINES.md).
// Static slice: only selected input cells are read and domain-checked (L7).
// @outputs: float[1512]
fn main(a: [Field; 9072]) -> pub [Field; 1512] {
    let mut result = [0; 1512];
    for i in 0..1512 { result[i] = noir_hw::checked(a[7560 + i * 1]); }
    result
}
