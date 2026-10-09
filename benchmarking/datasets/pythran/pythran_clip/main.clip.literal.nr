// pythran_clip__clip: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each abs occurrence is recomputed; complex magnitude/division retain the optimized Q32 arithmetic family. L4 masks the inactive limit1 input to prevent unused products overflowing. Inactive zero division is totalized and validity gated.
// @outputs: float[2 * N]
global N: u32 = 2; // = z_0
fn magnitude(x: [Field; 2]) -> Field {
    noir_hw::sqrt(noir_hw::mul(x[0], x[0]) + noir_hw::mul(x[1], x[1]))
}
fn limit1(x: [Field; 2], outer: bool) -> [Field; 2] {
    let x = if outer { x } else { [0, 0] };
    let tiny = noir_hw::lt(magnitude(x), 4295);
    let mag = magnitude(x);
    let den = noir_hw::mul(mag, mag);
    assert(!(outer & !tiny) | (den != 0));
    let mut unit = [0; 2];
    for part in 0..2 {
        let num = noir_hw::mul(x[part], mag);
        // Safe inactive divisor is arithmetic plumbing for L4's eagerly computed branches.
        let quotient = noir_hw::div(
            if den == 0 { 0 } else { num },
            if den == 0 { 1 } else { den },
        );
        unit[part] = if tiny { 0 } else { quotient };
    }
    unit
}

fn main(z: [[Field; 2]; N], _max: Field) -> pub [[Field; 2]; N] {
    noir_hw::check_input(_max);
    for i in 0..N {
        noir_hw::check_inputs(z[i]);
    }
    let mut result = [[0; 2]; N];
    for i in 0..N {
        let over = noir_hw::gt(magnitude(z[i]), _max);
        let unit = limit1(z[i], over);
        let limited = [noir_hw::mul(unit[0], _max), noir_hw::mul(unit[1], _max)];
        for part in 0..2 {
            result[i][part] = if over { limited[part] } else { z[i][part] };
        }
    }
    result
}
