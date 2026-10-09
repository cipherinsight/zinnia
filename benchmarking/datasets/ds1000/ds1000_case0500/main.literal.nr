// DS-1000 case 0500: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reshape changes only the static shape; row-major wires keep their order.
// @outputs: float[54]
fn main(a: [Field; 54]) -> pub [Field; 54] {
    noir_hw::check_inputs(a);
    // a = a.reshape(-1, 3)
    let a = a;
    a
}
