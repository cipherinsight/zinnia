// DS-1000 case 0363, literal translation of kernel.py (LITERAL_BASELINES.md).
// np.insert uses its definition: normalize pos in [-100,100], then select prefix / inserted value / shifted suffix for each output.
// @outputs: float[101]
fn main(a: [Field; 100], pos: Field, element: Field) -> pub [Field; 101] {
 noir_hw::check_inputs(a); noir_hw::check_input(pos);
 noir_hw::check_input(element);
 let elements = [noir_hw::from_int(element)];
 let point = if noir_hw::lt(pos, 0) { pos + 100 } else { pos };
 assert(noir_hw::ge(point, 0) & noir_hw::le(point, 100));
 let mut result = [0; 101];
 for k in 0..101 {
  let before = noir_hw::lt(k as Field, point);
  let equal = point == k as Field;
  for j in 0..1 {
   let left = if k < 100 { a[k*1+j] } else { 0 };
   let right = if k > 0 { a[(k-1)*1+j] } else { 0 };
   result[k*1+j] = if before { left } else { if equal { elements[j] } else { right } };
  }
 }
 result
}
