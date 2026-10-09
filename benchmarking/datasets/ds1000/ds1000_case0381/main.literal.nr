// DS-1000 case 0381, literal translation of kernel.py (LITERAL_BASELINES.md).
// Stable adjacent bubble sort uses n(n-1)/2 compare-and-swaps, with fresh bounds per comparison.
// @outputs: int[100]
fn main(a: [Field; 100]) -> pub [Field; 100] {
 noir_hw::check_inputs(a);
 let (_, sorted_indices) = bubble(a);
 let mut reversed = [0; 100];
 for i in 0..100 { reversed[i] = sorted_indices[99-i]; }
 let mut result = [0; 100];
 for i in 0..100 { result[i] = reversed[i]; }
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
