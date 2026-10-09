// DS-1000 case 0356, literal translation of kernel.py (LITERAL_BASELINES.md).
// Stable adjacent bubble sort uses n(n-1)/2 compare-and-swaps, with fresh bounds per comparison.
// @outputs: int[27]
fn main(a: [Field; 27], b: [Field; 27]) -> pub [Field; 27] {
 noir_hw::check_inputs(a); noir_hw::check_inputs(b);
 let mut sort_indices = [0; 27];
 for j in 0..3 { for k in 0..3 {
  let mut column = [0; 3];
  for i in 0..3 { column[i] = a[i*9+j*3+k]; }
  let (_, order) = bubble(column);
  for i in 0..3 { sort_indices[i*9+j*3+k] = order[i]; }
 }}
 let mut static_indices: [[u32; 27]; 3] = [[0; 27]; 3];
 for p in 0..27 { static_indices[0][p] = p/9; static_indices[1][p] = p/3%3; static_indices[2][p] = p%3; }
 let mut result = [0; 27];
 for p in 0..27 {
  let j = static_indices[1][p]; let k = static_indices[2][p];
  let mut column = [0; 3];
  for i in 0..3 { column[i] = b[i*9+j*3+k]; }
  result[p] = column[noir_hw::index_pos::<8>(sort_indices[p], 3)];
 }
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
