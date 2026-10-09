// DS-1000 case 0387, literal translation of kernel.py (LITERAL_BASELINES.md).
// Each reshape and swapaxes has its full intermediate wire array; constant permutations need no arithmetic.
// @outputs: float[20000]
fn main(a: [Field; 20000]) -> pub [Field; 20000] {
 noir_hw::check_inputs(a);
 let reshaped = a;
 let mut swapped = [0; 20000];
 for i in 0..50 { for j in 0..100 { for r in 0..2 { for s in 0..2 { swapped[((i*100+j)*2+r)*2+s] = reshaped[((i*2+r)*100+j)*2+s]; }}}}
 let result = swapped; // final reshape preserves storage
 result
}
