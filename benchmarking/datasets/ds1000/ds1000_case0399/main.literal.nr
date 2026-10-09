// DS-1000 case 0399, literal translation of kernel.py (LITERAL_BASELINES.md).
// Loop-index branches fold at compile time; integer times Q32 stays exact, with separate products and additions.
// @outputs: float[30]
fn main(A: [Field; 30], a: Field, b: Field, c: Field) -> pub [Field; 30] {
 noir_hw::check_inputs(A); noir_hw::check_input(a); noir_hw::check_input(b);
 noir_hw::check_input(c);
 let mut B = [0; 30];
 for k in 0..30 {
  if k == 0 { B[k] = a*A[k]; }
  else if k == 1 { B[k] = a*A[k] + b*B[k-1]; }
  else { B[k] = a*A[k] + b*B[k-1] + c*B[k-2]; }
 }
 B
}
