// DS-1000 case 0329, Noir baseline.
// Kernel: `a ** power` with the constant power = 6 on Float[8, 6].
// Square-and-multiply as in the halo2 baseline: x^2, x^4 = x^2 * x^2,
// x^2 * x^4, three rescaled products per element.
// @outputs: float[48]

use noir_hw::mul;

fn main(a: [Field; 48]) -> pub [Field; 48] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut powered = [0; 48];
    for k in 0..48 {
        let x2 = mul(a[k], a[k]);
        let x4 = mul(x2, x2);
        powered[k] = mul(x2, x4);
    }
    powered
}
