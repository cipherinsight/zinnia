// DS-1000 case 0393, Noir baseline.
// Kernel: `a[:, 4:min(7, 10)]` on Float[20, 10] (`low = 4`, `high = 7` are
// constants, and the clamp to a.shape[1] is a no-op).
// @outputs: float[60]

fn main(a: [Field; 200]) -> pub [Field; 60] {
    let mut out = [0; 60];
    for i in 0..20 {
        for j in 0..3 {
            out[i * 3 + j] = a[i * 10 + 4 + j];
        }
    }
    out
}
