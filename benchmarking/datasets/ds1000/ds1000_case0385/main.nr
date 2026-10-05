// DS-1000 case 0385, Noir baseline.
// Kernel: `a.reshape(50, 2, 100, 2).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)` on Float[100, 200]: the 2x2 blocks of a as patches;
// block (i, j) is output patch j * 50 + i.
// @outputs: float[20000]

fn main(a: [Field; 20000]) -> pub [Field; 20000] {
    let mut patches = [0; 20000];
    let mut p = 0;
    for j in 0..100 {
        for i in 0..50 {
            for r in 0..2 {
                for s in 0..2 {
                    patches[p] = a[(2 * i + r) * 200 + 2 * j + s];
                    p += 1;
                }
            }
        }
    }
    patches
}
