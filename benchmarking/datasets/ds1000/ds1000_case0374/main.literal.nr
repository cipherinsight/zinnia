// DS-1000 case 0374, literal translation of kernel.py (LITERAL_BASELINES.md).
// ecdf_result remains a helper: full stable bubble sort and constant-folded ys.
// Argmax scans (best value, first best index); branches compute both results then select.
// @outputs: float[5]
fn main(grades: [Field; 50], eval: [Field; 5]) -> pub [Field; 5] {
 noir_hw::check_inputs(grades); noir_hw::check_inputs(eval);
 let (resultx, resulty) = ecdf_result(grades);
 let mut result = [0; 5];
 for i in 0..5 {
  let element = eval[i];
  let below = noir_hw::lt(noir_hw::from_int(element), resultx[0]);
  let above = noir_hw::ge(noir_hw::from_int(element), resultx[49]);
  let e = noir_hw::from_int(element);
  let mut mask = [0; 50];
  for j in 0..50 { mask[j] = noir_hw::gt(resultx[j], e) as Field; }
  let at = argmax(mask);
  let mid = resulty[noir_hw::index_pos::<8>(at-1, 50)];
  result[i] = if below { 0 } else { if above { 4294967296 } else { mid } };
 }
 result
}
fn ecdf_result(x: [Field; 50]) -> ([Field; 50], [Field; 50]) {
 let mut xs = x;
 for pass in 0..49 { for j in 0..49-pass {
  let swap = noir_hw::gt(xs[j], xs[j+1]);
  let left = xs[j]; let right = xs[j+1];
  xs[j] = if swap { right } else { left };
  xs[j+1] = if swap { left } else { right };
 }}
 // Constant-folded arange(1,51)/float(50), quantized once.
 let ys: [Field; 50] = [85899346, 171798692, 257698038, 343597384, 429496730, 515396076, 601295421, 687194767, 773094113, 858993459, 944892805, 1030792151, 1116691497, 1202590843, 1288490189, 1374389535, 1460288881, 1546188227, 1632087572, 1717986918, 1803886264, 1889785610, 1975684956, 2061584302, 2147483648, 2233382994, 2319282340, 2405181686, 2491081032, 2576980378, 2662879724, 2748779069, 2834678415, 2920577761, 3006477107, 3092376453, 3178275799, 3264175145, 3350074491, 3435973837, 3521873183, 3607772529, 3693671875, 3779571220, 3865470566, 3951369912, 4037269258, 4123168604, 4209067950, 4294967296];
 (xs, ys)
}
fn argmax(values: [Field; 50]) -> Field {
 let mut best = values[0]; let mut index: Field = 0;
 for i in 1..50 {
  let greater = noir_hw::gt(values[i], best);
  best = if greater { values[i] } else { best };
  index = if greater { i as Field } else { index };
 }
 index
}
