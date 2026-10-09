// DS-1000 case 0358, literal translation of kernel.py (LITERAL_BASELINES.md).
// Stable adjacent bubble sort uses n(n-1)/2 compare-and-swaps, with fresh bounds per comparison.
// @outputs: int[27]
fn main(a: [Field; 27], b: [Field; 27]) -> pub [Field; 27] {
 noir_hw::check_inputs(a); noir_hw::check_inputs(b);
 let mut sums = [0; 3];
 for i in 0..3 {
  let mut sum = a[i*9]; for j in 1..9 { sum += a[i*9+j]; } sums[i] = sum;
 }
 let (_, index) = bubble(sums);
 let mut result = [0; 27];
 for i in 0..3 { for j in 0..9 {
  let mut column = [0; 3]; for k in 0..3 { column[k] = b[k*9+j]; }
  result[i*9+j] = column[noir_hw::index_pos::<8>(index[i], 3)];
 }}
 result
}
fn bubble<let N: u32>(values: [Field; N]) -> ([Field; N], [Field; N]) {
 let mut keys = values;
 let mut indices = [0; N];
 for i in 0..N { indices[i] = i as Field; }
 for pass in 0..N-1 { for j in 0..N-1-pass {
  let swap = noir_hw::gt(keys[j], keys[j+1]);
  let left = keys[j]; let right = keys[j+1];
  keys[j] = if swap { right } else { left };
  keys[j+1] = if swap { left } else { right };
  let left = indices[j]; let right = indices[j+1];
  indices[j] = if swap { right } else { left };
  indices[j+1] = if swap { left } else { right };
 }}
 (keys, indices)
}
