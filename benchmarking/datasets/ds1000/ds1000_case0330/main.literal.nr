// DS-1000 case 0330: literal translation of kernel.py (LITERAL_BASELINES.md).
// Local power operation uses the shared Q32 products in the same repeated-squaring order as pow_const.
// @outputs: float[48]
fn power_six(x: Field) -> Field {
    let x2 = noir_hw::mul(x, x);
    let x4 = noir_hw::mul(x2, x2);
    noir_hw::mul(x2, x4)
}
fn main(a: [Field; 48]) -> pub [Field; 48] {
    noir_hw::check_inputs(a);
    let mut result = [0; 48];
    for i in 0..48 { result[i] = power_six(a[i]); }
    result
}
