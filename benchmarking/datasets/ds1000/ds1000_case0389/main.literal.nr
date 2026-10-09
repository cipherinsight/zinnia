// DS-1000 case 0389, literal translation of kernel.py (LITERAL_BASELINES.md).
// Each reshape, swapaxes and transpose has its full intermediate wire array; constant permutations need no arithmetic.
// @outputs: float[72]
fn main(a: [Field; 72]) -> pub [Field; 72] {
 noir_hw::check_inputs(a);
 let reshaped = a;
 let mut swapped = [0; 72];
 for i in 0..2 { for r in 0..3 { for j in 0..3 { for s in 0..4 { swapped[((i*3+r)*3+j)*4+s] = reshaped[((i*3+j)*3+r)*4+s]; }}}}
 let result = swapped;
 result
}
