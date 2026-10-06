// DS-1000 case 0336, Noir baseline.
// Kernel: `np.diag(np.fliplr(a))` on Float[8, 8]: the anti-diagonal a[i, 7 - i].
// @outputs: float[8]

fn main(a: [Field; 64]) -> pub [Field; 8] {
    let mut anti_diag = [0; 8];
    for i in 0..8 {
        anti_diag[i] = noir_hw::checked(a[i * 8 + 7 - i]);
    }
    anti_diag
}
