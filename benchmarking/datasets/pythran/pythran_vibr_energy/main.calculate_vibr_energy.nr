// pythran_vibr_energy__calculate_vibr_energy (Pythran vibr_energy), Noir baseline.
// Kernel: `numpy.exp(-harmonic * i - anharmonic * i ** 2)` on harmonic,
// anharmonic, i: Float[N]. In NumPy's order: (-harmonic) * i (the negation is
// exact, so the product rounds -(harmonic * i) down), i ** 2 and
// anharmonic * i ** 2 are rounded products, the difference is exact, and
// `exp` (the shared gadget, NPB_BASELINES.md section 5, rule 5) checks it to
// the window.
// @outputs: float[N]

use noir_hw::{check_inputs, exp, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = harmonic_0

fn main(harmonic: [Field; N], anharmonic: [Field; N], i: [Field; N]) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(harmonic);
    check_inputs(anharmonic);
    check_inputs(i);
    let mut energies: [Field; N] = [0; N];
    for k in 0..N {
        let exponent = mul(-harmonic[k], i[k]) - mul(anharmonic[k], mul(i[k], i[k]));
        energies[k] = exp(exponent);
    }
    energies
}
