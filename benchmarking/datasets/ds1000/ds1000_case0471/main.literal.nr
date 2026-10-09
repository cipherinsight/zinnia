// DS-1000 case 0471: literal translation of kernel.py (LITERAL_BASELINES.md).
// Static slice: only selected input cells are read and domain-checked (L7).
// @outputs: float[2268]
fn main(Z: [Field; 9072]) -> pub [Field; 2268] {
    let mut result = [0; 2268];
    for i in 0..2268 { result[i] = noir_hw::checked(Z[3 + i * 4]); }
    result
}
