// DS-1000 case 0337, Noir baseline.
// Kernel: `np.diag(np.fliplr(a))` on Float[8, 10]: the anti-diagonal a[i, 9 - i].
// @outputs: float[8]

fn main(a: [Field; 80]) -> pub [Field; 8] {
    let mut anti_diag = [0; 8];
    for i in 0..8 {
        anti_diag[i] = noir_hw::checked(a[i * 10 + 9 - i]);
    }
    anti_diag
}
