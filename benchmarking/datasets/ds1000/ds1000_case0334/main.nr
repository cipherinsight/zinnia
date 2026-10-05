// DS-1000 case 0334, Noir baseline.
// Kernel: `np.mean([a, b, c], axis=0)` on three Float[50]. The Q32 mean
// floor((a + b + c) * 2^32 / (3 * 2^32)) equals floor((a + b + c) / 3).
// @outputs: float[50]

use noir_hw::divmod_pos;

fn main(a: [Field; 50], b: [Field; 50], c: [Field; 50]) -> pub [Field; 50] {
    let mut mean = [0; 50];
    for i in 0..50 {
        mean[i] = divmod_pos(a[i] + b[i] + c[i], 3).0;
    }
    mean
}
