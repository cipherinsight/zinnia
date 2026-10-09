// pythran_clip__clip (Pythran clip), Noir baseline.
// Kernel: for each z[i] of z: Complex[N], limit1(z[i]) * _max when
// abs(z[i]) > _max, else z[i], with limit1(x) = 0 when abs(x) < 1e-6 and
// x / abs(x) otherwise; returns Complex[N].
// Q32 component squares precede sqrt. Complex/real division keeps rounded
// x*d and d*d before division, matching the frozen primitive schedule.
// Retain inactive products. Zero-division maps to zero with a guarded source assertion.
// @outputs: float[2 * N]

use noir_hw::{check_input, check_inputs, divmod_pos, sqrt, mul, window, SCALE};

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
        let magnitude = sqrt(mul(x[0], x[0]) + mul(x[1], x[1]));
        // `sqrt` proved magnitude < 2^63, so its window is magnitude + 2^63.
        let magnitude_w = window(magnitude);
        let over = magnitude_w > max_w;
        let tiny = magnitude_w < epsilon_w;
        let active = over & !tiny;
        let denominator = mul(magnitude, magnitude);
        assert(!active | (denominator != 0), "active complex division by zero");
        let zero = denominator == 0;
        let safe_denominator = denominator + zero as Field;
        for part in 0..2 {
            let numerator = mul(x[part], magnitude);
            let (unit, _) = divmod_pos((if zero { 0 } else { numerator }) * SCALE, safe_denominator);
            let unit = if active { unit } else { 0 };
            let limited = mul(unit, _max);
            result[i][part] = if over { limited } else { x[part] };
        }
    }
    result
}
