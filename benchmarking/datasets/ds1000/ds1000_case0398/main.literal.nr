// DS-1000 case 0398, literal translation of kernel.py (LITERAL_BASELINES.md).
// Loop-index branches fold at compile time; integer times Q32 stays exact, with separate products and additions.
// @outputs: float[30]
fn main(A: [Field; 30], a: Field, b: Field) -> pub [Field; 30] {
 noir_hw::check_inputs(A); noir_hw::check_input(a); noir_hw::check_input(b);
 let mut B = [0; 30];
 for k in 0..30 {
  if k == 0 { B[k] = a*A[k]; }
  else { B[k] = a*A[k] + b*B[k-1]; }
 }
 B
}
