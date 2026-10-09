// pythran_crotg__CROTG: literal translation of kernel.py (LITERAL_BASELINES.md).
// Repeated abs calls remain separate. Complex primitives retain current Q32 rounding. Inactive zero divisions are totalized and nonzero assertions guarded; outputs select both branches.
// @outputs: float[7]

use noir_hw::{check_inputs, mul, SCALE, sqrt};

fn magnitude(x: [Field; 2]) -> Field {
    sqrt(mul(x[0], x[0]) + mul(x[1], x[1]))
}
fn divide(n: Field, d: Field, active: bool) -> Field {
    let zero = d == 0;
    assert(!active | !zero);
    noir_hw::canonical_div_nonzero(if zero { 0 } else { n }, d + zero as Field)
}
fn complex_div(x: [Field; 2], d: Field, active: bool) -> [Field; 2] {
    let denominator = mul(d, d);
    [divide(mul(x[0], d), denominator, active), divide(mul(x[1], d), denominator, active)]
}
fn main(CA: [Field; 2], CB: [Field; 2]) -> pub ([Field; 2], [Field; 2], Field, [Field; 2]) {
    check_inputs(CA);
    check_inputs(CB);
    let zero = magnitude(CA) == 0;
    let active = !zero;
    let scale = magnitude(CA) + magnitude(CB);
    let a_magnitude = magnitude(complex_div(CA, scale, active));
    let a_squared = mul(a_magnitude, a_magnitude);
    let b_magnitude = magnitude(complex_div(CB, scale, active));
    let b_squared = mul(b_magnitude, b_magnitude);
    let norm = mul(scale, sqrt(a_squared + b_squared));
    let alpha = complex_div(CA, magnitude(CA), active);
    let c = divide(magnitude(CA), norm, active);
    let conjugate = [CB[0], -CB[1]];
    let product = [
        mul(alpha[0], conjugate[0]) - mul(alpha[1], conjugate[1]),
        mul(alpha[0], conjugate[1]) + mul(alpha[1], conjugate[0]),
    ];
    let s = complex_div(product, norm, active);
    let ca = [mul(alpha[0], norm), mul(alpha[1], norm)];
    (if zero { CB } else { ca }, CB, if zero { 0 } else { c }, if zero { [SCALE, 0] } else { s })
}
