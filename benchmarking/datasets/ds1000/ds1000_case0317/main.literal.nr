// DS-1000 case 0317: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reshape/copy is a complete array of wire aliases, with no arithmetic.
// @outputs: float[1000]
fn main(a: [Field; 1000]) -> pub [Field; 1000] {
    noir_hw::check_inputs(a);
    let result = a;
    result
}
