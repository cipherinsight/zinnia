// DS-1000 case 0364, literal translation of kernel.py (LITERAL_BASELINES.md).
// np.insert uses its definition: normalize pos in [-100,100], then select prefix / inserted value / shifted suffix for each output.
// @outputs: float[1010]
fn main(a: [Field; 1000], pos: Field, element: [Field; 10]) -> pub [Field; 1010] {
 noir_hw::check_inputs(a); noir_hw::check_input(pos);
 noir_hw::check_inputs(element);
 let elements = element;
 let point = if noir_hw::lt(pos, 0) { pos + 100 } else { pos };
 assert(noir_hw::ge(point, 0) & noir_hw::le(point, 100));
 let mut result = [0; 1010];
 for k in 0..101 {
  let before = noir_hw::lt(k as Field, point);
  let equal = point == k as Field;
  for j in 0..10 {
   let left = if k < 100 { a[k*10+j] } else { 0 };
   let right = if k > 0 { a[(k-1)*10+j] } else { 0 };
   result[k*10+j] = if before { left } else { if equal { elements[j] } else { right } };
  }
 }
 result
}
