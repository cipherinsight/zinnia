// DS-1000 case 0491: literal translation of kernel.py (LITERAL_BASELINES.md).
// isclose uses abs(a-b) <= atol + rtol*abs(b); defaults quantize to 43 and 42950. Full boolean arrays precede sum.
// @outputs: int[1]
fn main(s1: [Field; 6], s2: [Field; 6]) -> pub Field {
    noir_hw::check_inputs(s1); noir_hw::check_inputs(s2);
    let mut close = [false; 6];
    for i in 0..6 {
        let d = noir_hw::abs_int64(s1[i]-s2[i]);
        let ab = noir_hw::abs_int64(s2[i]);
        let tol = 43 + noir_hw::mul(42950, ab);
        close[i] = noir_hw::le(d, tol);
    }
    let mut not_close = [false; 6];
    for i in 0..6 { not_close[i] = !close[i]; }
    let mut result: Field = 0;
    for i in 0..6 { result += not_close[i] as Field; }
    result
}
