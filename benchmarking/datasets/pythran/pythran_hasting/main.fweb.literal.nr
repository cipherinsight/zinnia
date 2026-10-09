// pythran_hasting__fweb: literal translation of kernel.py (LITERAL_BASELINES.md).
// Repeated predation expressions are recomputed at every occurrence. Unread t is omitted; canonical nonzero division matches optimized arithmetic.
// @outputs: float[3]

use noir_hw::{canonical_div_nonzero, check_input, checked, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global Y_LEN: u32 = 3; // = y_0

// a * u * v / (1.0 + b * u)

fn main(
    y: [Field; Y_LEN],
    a1: Field,
    a2: Field,
    b1: Field,
    b2: Field,
    d1: Field,
    d2: Field,
) -> pub [Field; 3] {
    let y = [checked(y[0]), checked(y[1]), checked(y[2])];
    check_input(a1);
    check_input(a2);
    check_input(b1);
    check_input(b2);
    check_input(d1);
    check_input(d2);
    let mut yprime = [0; 3];
    yprime[0] = mul(y[0], SCALE - y[0])
        - canonical_div_nonzero(mul(mul(a1, y[0]), y[1]), SCALE + mul(b1, y[0]));
    yprime[1] = canonical_div_nonzero(mul(mul(a1, y[0]), y[1]), SCALE + mul(b1, y[0]))
        - canonical_div_nonzero(mul(mul(a2, y[1]), y[2]), SCALE + mul(b2, y[1]))
        - mul(d1, y[1]);
    yprime[2] =
        canonical_div_nonzero(mul(mul(a2, y[1]), y[2]), SCALE + mul(b2, y[1])) - mul(d2, y[2]);
    yprime
}
