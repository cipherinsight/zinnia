// DS-1000 case 0409, Noir baseline.
// Kernel: `np.array(x) + np.array(y)` on two Float[8, 6] (the copies are no-ops).
// @outputs: float[48]

fn main(x: [Field; 48], y: [Field; 48]) -> pub [Field; 48] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(x);
    noir_hw::check_inputs(y);
    let mut z = [0; 48];
    for i in 0..48 {
        z[i] = x[i] + y[i];
    }
    z
}
