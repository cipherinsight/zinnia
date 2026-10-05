// DS-1000 case 0402, Noir baseline.
// Kernel: `np.ravel_multi_index(index, dims=(8, 9, 8, 8, 8), order='F')` on a
// private Integer[5] index; the Float[8, 9, 8, 8, 8] parameter a is never read
// by the kernel and enters no constraint. NumPy raises unless
// 0 <= index[k] < dims[k]: a 3-bit range check proves it for a dim of 8, two
// 4-bit checks (index and 8 - index) for the dim of 9. The flat index uses the
// constant Fortran strides (1, 8, 72, 576, 4608).
// @outputs: int[1]

fn main(a: [Field; 36864], index: [Field; 5]) -> pub Field {
    let _ = a;
    index[0].assert_max_bit_size::<3>();
    index[1].assert_max_bit_size::<4>();
    (8 - index[1]).assert_max_bit_size::<4>();
    index[2].assert_max_bit_size::<3>();
    index[3].assert_max_bit_size::<3>();
    index[4].assert_max_bit_size::<3>();
    index[0] + 8 * index[1] + 72 * index[2] + 576 * index[3] + 4608 * index[4]
}
