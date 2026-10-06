// DS-1000 case 0403, Noir baseline.
// Kernel: `np.ravel_multi_index(index, dims=(8, 9, 8, 8, 8), order='C')` on a
// private Integer[5] index; the Float[8, 9, 8, 8, 8] parameter a is never read
// by the kernel and enters no constraint. NumPy raises unless
// 0 <= index[k] < dims[k]: a 3-bit range check proves it for a dim of 8, two
// 4-bit checks (index and 8 - index) for the dim of 9. The flat index uses the
// constant C strides (4608, 512, 64, 8, 1).
// These bounds also imply the input domain (an int64), so index needs no
// separate domain check; a is never read and gets none.
// @outputs: int[1]

fn main(a: [Field; 36864], index: [Field; 5]) -> pub Field {
    let _ = a;
    index[0].assert_max_bit_size::<3>();
    index[1].assert_max_bit_size::<4>();
    (8 - index[1]).assert_max_bit_size::<4>();
    index[2].assert_max_bit_size::<3>();
    index[3].assert_max_bit_size::<3>();
    index[4].assert_max_bit_size::<3>();
    4608 * index[0] + 512 * index[1] + 64 * index[2] + 8 * index[3] + index[4]
}
