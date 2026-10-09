// DS-1000 case 0391, literal translation of kernel.py (LITERAL_BASELINES.md).
// Constant slicing reads only columns 4, 5 and 6; each read input is checked once.
// @outputs: float[60]
fn main(a: [Field; 200]) -> pub [Field; 60] {
 let low = 4; let high = 7;
 let mut result = [0; 60];
 for i in 0..20 { for j in low..high { result[i*3+j-low] = noir_hw::checked(a[i*10+j]); }}
 result
}
