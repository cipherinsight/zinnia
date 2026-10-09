// DS-1000 case 0497: literal translation of kernel.py (LITERAL_BASELINES.md).
// Pad materializes all output slots; the scalar fill is converted once to the array dtype.
// @outputs: float[1209]
fn main(a: [Field; 420], element: Field) -> pub [Field; 1209] {
    noir_hw::check_inputs(a);
    noir_hw::check_input(element);
    let fill = noir_hw::from_int(element);
    let mut result = [fill; 1209];
    for i in 0..42 { for j in 0..10 { result[i*13+j] = a[i*10+j]; } }
    result
}
