// DS-1000 case 0403, Noir LITERAL baseline (LITERAL_BASELINES.md).
// Literal translation of kernel.py:
//   dims = np.array([8, 9, 8, 8, 8])
//   result = np.ravel_multi_index(index, dims=dims, order='C')
// np.ravel_multi_index has no source to copy (L5): the textbook definition is
// the defining sum flat = sum_k index[k] * stride[k] with the C-order strides
// (4608, 512, 64, 8, 1), accumulated term by term in index order k = 0..5.
// NumPy raises unless 0 <= index[k] < dims[k]; this is also exactly the
// input-domain check of CONVENTIONS.md 1.6 for an int64 (a check the
// kernel's own semantics already impose and that implies the domain replaces
// the generic one, as 1.6 says by name for this case: "0402/0403"). A dim of
// 8 is one bit-size assertion; a dim of 9 is not a power of two and needs the
// standard two-sided Noir technique (bound above and below 2^4, since Noir's
// stdlib has no less-than-constant primitive other than assert_max_bit_size):
// this is the plumbing of CONVENTIONS.md 1.6, not a computational shortcut,
// so it is copied from the optimized baseline (L7).
// `a` (the Float[8,9,8,8,8] parameter) is never read by the kernel, so it
// enters no constraint and gets no check.
// @outputs: int[1]

fn main(a: [Field; 36864], index: [Field; 5]) -> pub Field {
    let _ = a;
    index[0].assert_max_bit_size::<3>();
    index[1].assert_max_bit_size::<4>();
    (8 - index[1]).assert_max_bit_size::<4>();
    index[2].assert_max_bit_size::<3>();
    index[3].assert_max_bit_size::<3>();
    index[4].assert_max_bit_size::<3>();

    // flat = sum_k index[k] * stride[k], one term added at a time, in source
    // (index) order.
    let mut flat: Field = 0;
    flat = flat + 4608 * index[0];
    flat = flat + 512 * index[1];
    flat = flat + 64 * index[2];
    flat = flat + 8 * index[3];
    flat = flat + index[4];
    flat
}
