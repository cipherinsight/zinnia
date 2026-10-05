// DS-1000 case 0391, Noir baseline.
// Kernel: `a[:, 4:7]` on Float[20, 10] (`low = 4`, `high = 7` are constants).
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
