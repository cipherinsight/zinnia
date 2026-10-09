// pythran_clip__clip (Pythran clip), Noir baseline.
// Kernel: for each z[i] of z: Complex[N], limit1(z[i]) * _max when
// abs(z[i]) > _max, else z[i], with limit1(x) = 0 when abs(x) < 1e-6 and
// x / abs(x) otherwise; returns Complex[N].
// abs of a complex value is the exact floor square root of re^2 + im^2
// (`isqrt` of the raw sum, scale 2^64): one rounding, as np.abs rounds hypot
// once. Both branches are computed and selected; in the unused one the
// divisor is moved off zero (abs + [abs < 1e-6]), so every input has a
// witness. x / abs(x) and * _max are componentwise, as NumPy divides and
// multiplies a complex value by a real one.
// @outputs: float[2 * N]

use noir_hw::{check_input, check_inputs, divmod_pos, isqrt, mul, window, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 2; // = z_0

global EPSILON_Q: Field = 4295; // round(1e-6 * 2^32)

fn main(z: [[Field; 2]; N], _max: Field) -> pub [[Field; 2]; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(_max);
    let max_w = window(_max);
    let epsilon_w = window(EPSILON_Q);
    let mut result: [[Field; 2]; N] = [[0; 2]; N];
    for i in 0..N {
        check_inputs(z[i]);
        let x = z[i];
        let magnitude = isqrt(x[0] * x[0] + x[1] * x[1]);
        // `isqrt` proved magnitude < 2^63, so its window is magnitude + 2^63.
        let magnitude_w = window(magnitude);
        let over = magnitude_w > max_w;
        let tiny = magnitude_w < epsilon_w;
        let divisor = magnitude + tiny as Field;
        for part in 0..2 {
            let (unit, _) = divmod_pos(x[part] * SCALE, divisor);
            let limited = if tiny { 0 } else { mul(unit, _max) };
            result[i][part] = if over { limited } else { x[part] };
        }
    }
    result
}
