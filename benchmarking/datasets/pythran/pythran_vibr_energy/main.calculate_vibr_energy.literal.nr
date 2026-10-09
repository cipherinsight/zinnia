// pythran_vibr_energy__calculate_vibr_energy: literal translation of kernel.py (LITERAL_BASELINES.md).
// Full elementwise intermediates, negation before multiplication. Exp uses the same canonical shared gadget as optimized.
// @outputs: float[N]

use noir_hw::{check_inputs, canonical_exp as exp, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = harmonic_0

fn main(harmonic: [Field; N], anharmonic: [Field; N], i: [Field; N]) -> pub [Field; N] {
    check_inputs(harmonic); check_inputs(anharmonic); check_inputs(i);
    let mut negative = [0; N]; for k in 0..N { negative[k] = -harmonic[k]; }
    let mut linear = [0; N]; for k in 0..N { linear[k] = mul(negative[k], i[k]); }
    let mut square = [0; N]; for k in 0..N { square[k] = mul(i[k], i[k]); }
    let mut quadratic = [0; N]; for k in 0..N { quadratic[k] = mul(anharmonic[k], square[k]); }
    let mut exponent = [0; N]; for k in 0..N { exponent[k] = linear[k] - quadratic[k]; }
    let mut energies = [0; N]; for k in 0..N { energies[k] = exp(exponent[k]); }
    energies
}
