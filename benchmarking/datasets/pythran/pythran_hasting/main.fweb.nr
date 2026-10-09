// pythran_hasting__fweb (Pythran hasting), Noir baseline.
// Kernel: the right-hand side of the Hastings food-web ODE for `y: Float[3]`
// and six scalar rates; `t` is never read (not loaded), nor are the elements
// of y after the third (none at the fixture's shape).
// Every product and quotient is rounded in Python's left-to-right order
// (a1 * y[0] * y[1] / (1.0 + b1 * y[0]) is ((a1 * y0) * y1) / (..)). The two
// predation terms a1 y0 y1 / (1 + b1 y0) and a2 y1 y2 / (1 + b2 y1) appear
// twice each in the kernel with the same operands, so each is computed once.
// The divisors can have either sign (`div`).
// @outputs: float[3]

use noir_hw::{check_input, checked, canonical_div_nonzero, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global Y_LEN: u32 = 3; // = y_0

// a * u * v / (1.0 + b * u)
fn predation(a: Field, u: Field, v: Field, b: Field) -> Field {
    canonical_div_nonzero(mul(mul(a, u), v), SCALE + mul(b, u))
}

fn main(y: [Field; Y_LEN], a1: Field, a2: Field, b1: Field, b2: Field, d1: Field, d2: Field) -> pub [Field; 3] {
    // Input domain: every input element read is a Q32 code in the window.
    let y = [checked(y[0]), checked(y[1]), checked(y[2])];
    check_input(a1);
    check_input(a2);
    check_input(b1);
    check_input(b2);
    check_input(d1);
    check_input(d2);
    let growth = mul(y[0], SCALE - y[0]);
    let predation1 = predation(a1, y[0], y[1], b1);
    let predation2 = predation(a2, y[1], y[2], b2);
    [
        growth - predation1,
        predation1 - predation2 - mul(d1, y[1]),
        predation2 - mul(d2, y[2]),
    ]
}
