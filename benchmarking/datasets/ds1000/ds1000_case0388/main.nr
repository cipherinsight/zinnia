// DS-1000 case 0388, Noir baseline.
// Kernel: crop a to multiples of `patch_size = 5` (a constant; 100 x 200 already
// are), then `reshape(20, 5, 40, 5).swapaxes(1, 2).reshape(-1, 5, 5)` on
// Float[100, 200]: the 5x5 blocks of a as patches, block (i, j) is output patch
// i * 40 + j.
// @outputs: float[20000]

fn main(a: [Field; 20000]) -> pub [Field; 20000] {
    let mut patches = [0; 20000];
    let mut p = 0;
    for i in 0..20 {
        for j in 0..40 {
            for r in 0..5 {
                for s in 0..5 {
                    patches[p] = a[(5 * i + r) * 200 + 5 * j + s];
                    p += 1;
                }
            }
        }
    }
    patches
}
