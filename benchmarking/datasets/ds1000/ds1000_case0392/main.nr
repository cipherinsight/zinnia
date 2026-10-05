// DS-1000 case 0392, Noir baseline.
// Kernel: `a[4:7, :]` on Float[20, 10] (`low = 4`, `high = 7` are constants).
// @outputs: float[30]

fn main(a: [Field; 200]) -> pub [Field; 30] {
    let mut out = [0; 30];
    for k in 0..30 {
        out[k] = a[40 + k];
    }
    out
}
