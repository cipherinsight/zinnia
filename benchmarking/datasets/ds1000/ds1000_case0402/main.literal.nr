// DS-1000 case 0402, literal translation of kernel.py (LITERAL_BASELINES.md).
// Fortran ravel is an index-order sum of coordinate times constant stride; coordinate bounds reject NumPy errors. Parameter a is unread.
// @outputs: int[1]
fn main(a: [Field; 36864], index: [Field; 5]) -> pub Field {
 let _ = a;
 // Coordinate bounds imply int64 input domain, matching the optimized plumbing.
 index[0].assert_max_bit_size::<3>();
 index[1].assert_max_bit_size::<4>(); (8-index[1]).assert_max_bit_size::<4>();
 index[2].assert_max_bit_size::<3>(); index[3].assert_max_bit_size::<3>(); index[4].assert_max_bit_size::<3>();
 let dims = [8, 9, 8, 8, 8];
 let mut stride: Field = 1;
 let mut result: Field = 0;
 for k in 0..5 { result += index[k]*stride; stride *= dims[k]; }
 result
}
