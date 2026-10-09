// pythran_lap__laplacian (Pythran lap), Noir baseline.
// Kernel: the 4th-order periodic 1-D Laplacian of var: Float[N] built by nine
// slice updates of `lap` (with the kernel's quirk lap[0] = 4/3 * var[1], not
// var[-1]), then lap / dh2, under dh2 != 0.0 (`@requires`; the division has
// no witness for a zero divisor).
// Each update adds a constant times a slice of var; NumPy materializes the
// products, so each is rounded (`mul`), and a product of the same constant and
// element is computed once. The divisor's sign test is hoisted out of the
// elementwise division.
// @outputs: float[N]

use noir_hw::{check_input, check_inputs, divmod_pos, is_neg, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = var_0

global NEAR: Field = 5726623061; // round(4/3 * 2^32)
global CENTER: Field = -10737418240; // round(-5/2 * 2^32)
global FAR: Field = -357913941; // round(-1/12 * 2^32)

fn main(var: [Field; N], dh2: Field) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(var);
    check_input(dh2);
    let mut near: [Field; N] = [0; N];
    let mut center: [Field; N] = [0; N];
    let mut far: [Field; N] = [0; N];
    for i in 0..N {
        near[i] = mul(var[i], NEAR);
        center[i] = mul(var[i], CENTER);
        far[i] = mul(var[i], FAR);
    }
    let mut lap: [Field; N] = [0; N];
    for i in 1..N {
        lap[i] = near[i - 1]; // lap[1:] = 4/3 * var[:-1]
    }
    lap[0] = near[1]; // lap[0] = 4/3 * var[1]
    for i in 0..N - 1 {
        lap[i] += near[i + 1]; // lap[:-1] += 4/3 * var[1:]
    }
    lap[N - 1] += near[0]; // lap[-1] += 4/3 * var[0]
    for i in 0..N {
        lap[i] += center[i]; // lap += -5/2 * var
    }
    for i in 2..N {
        lap[i] += far[i - 2]; // lap[2:] += -1/12 * var[:-2]
    }
    for i in 0..2 {
        lap[i] += far[N - 2 + i]; // lap[:2] += -1/12 * var[-2:]
    }
    for i in 0..N - 2 {
        lap[i] += far[i + 2]; // lap[:-2] += -1/12 * var[2:]
    }
    for i in 0..2 {
        lap[N - 2 + i] += far[i]; // lap[-2:] += -1/12 * var[:2]
    }
    // lap / dh2 = floor(lap * 2^32 / dh2): divide by |dh2|, then restore the
    // sign (as noir_hw::div, with the sign test once).
    let negative = is_neg(dh2);
    let magnitude = if negative { -dh2 } else { dh2 };
    let mut result: [Field; N] = [0; N];
    for i in 0..N {
        let (quotient, _) = divmod_pos(lap[i] * SCALE, magnitude);
        result[i] = if negative { -quotient } else { quotient };
    }
    result
}
