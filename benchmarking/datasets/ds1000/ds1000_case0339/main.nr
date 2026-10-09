// DS-1000 case 0339, Noir baseline.
// Kernel: `np.vstack((np.diag(b), np.diag(np.fliplr(b))))` on Float[8, 10]:
// b = a[:8, :8]; row 0 is a[i, i], row 1 is a[i, 7 - i].
// @outputs: float[16]

fn main(a: [Field; 80]) -> pub [Field; 16] {
    let mut rows = [0; 16];
    for i in 0..8 {
        rows[i] = noir_hw::checked(a[i * 10 + i]);
        rows[8 + i] = noir_hw::checked(a[i * 10 + 7 - i]);
    }
    rows
}
