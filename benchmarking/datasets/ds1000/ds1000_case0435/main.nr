// DS-1000 case 0435, Noir baseline.
// Kernel: `a[1, :] = 0; a[:, 0] = 0; return a` on Integer[4, 4].
// The zeroed cells (row 1 and column 0) are circuit constants, not public
// values; the 9 cells that keep their input value (a[i, 1:] for rows 0, 2, 3)
// are returned in row-major order between the constant segments.
// @outputs: const int[1] = [0]; int[3]; const int[5] = [0, 0, 0, 0, 0]; int[3]; const int[1] = [0]; int[3]

fn main(a: [Field; 16]) -> pub [Field; 9] {
    [a[1], a[2], a[3], a[9], a[10], a[11], a[13], a[14], a[15]]
}
