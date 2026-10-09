// DS-1000 case 0370, literal translation of kernel.py (LITERAL_BASELINES.md).
// Full isclose broadcast, including self comparisons; tolerance recomputed per element, then running all reduction.
// @outputs: bool[1]
fn main(a: [Field; 9]) -> pub bool {
 noir_hw::check_inputs(a);
 let mut close = [false; 9];
 for i in 0..3 { for j in 0..3 {
  let y = a[j];
  let gap = noir_hw::abs_int64(a[i*3+j]-y);
  let tolerance = 0 + noir_hw::mul(42950, noir_hw::abs_int64(y));
  close[i*3+j] = noir_hw::le(gap, tolerance);
 }}
 let mut result = close[0];
 for k in 1..9 { result = result & close[k]; }
 result
}
