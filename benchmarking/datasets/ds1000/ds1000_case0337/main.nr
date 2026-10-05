// DS-1000 case 0337, Noir baseline.
// Kernel: `np.diag(np.fliplr(a))` on Float[8, 10]: fliplr(a)[i, j] = a[i, 9 - j],
// so the diagonal is a[i, 9 - i] for i < 8.
// @outputs: float[8]

fn main(a: [Field; 80]) -> pub [Field; 8] {
    let mut diag = [0; 8];
    for i in 0..8 {
        diag[i] = a[i * 10 + 9 - i];
    }
    diag
}
