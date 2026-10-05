// DS-1000 case 0478, Noir baseline.
// Kernel: result = np.einsum('ii->i', a); save = result.copy(); a[...] = 0;
// result[...] = save; return a, on Integer[4, 4].
// np.einsum('ii->i', a) is a writeable view of a's diagonal, so the last
// assignment writes the diagonal back into the zeroed a: the output is the
// diagonal of a with zeros elsewhere. The zeros are circuit constants, not
// public values; the 4 diagonal cells are returned between them.
// @outputs: int[1]; const int[4] = [0, 0, 0, 0]; int[1]; const int[4] = [0, 0, 0, 0]; int[1]; const int[4] = [0, 0, 0, 0]; int[1]

fn main(a: [Field; 16]) -> pub [Field; 4] {
    [a[0], a[5], a[10], a[15]]
}
