// pythran_ln__ln: literal translation of kernel.py (LITERAL_BASELINES.md).
// Local hint-free power helper preserves optimized MSB-first Q32 schedule; no powers or X-1 arrays shared.
// @outputs: float[N]

use noir_hw::{check_inputs, floor_div_const, floor_div_pow2, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = X_0

fn main(X: [Field; N], Y: [Field; N]) -> pub [Field; N] {
    // Input domain: every element of X is a Q32 code in the window.
    check_inputs(X);
    let mut result = [0; N];
    for i in 0..N { result[i] = X[i]-SCALE; }
    // (X - 1) ** 2 / 2
    let mut delta = [0; N];
    for i in 0..N { delta[i] = X[i]-SCALE; }
    let mut power = [0; N];
    for i in 0..N { power[i] = power_msb(delta[i],2); }
    let mut term = [0; N];
    for i in 0..N { term[i] = floor_div_const::<2>(power[i]); }
    for i in 0..N { result[i] -= term[i]; }
    // (X - 1) ** 3 / 3
    let mut delta = [0; N];
    for i in 0..N { delta[i] = X[i]-SCALE; }
    let mut power = [0; N];
    for i in 0..N { power[i] = power_msb(delta[i],3); }
    let mut term = [0; N];
    for i in 0..N { term[i] = floor_div_const::<3>(power[i]); }
    for i in 0..N { result[i] += term[i]; }
    // (X - 1) ** 4 / 4
    let mut delta = [0; N];
    for i in 0..N { delta[i] = X[i]-SCALE; }
    let mut power = [0; N];
    for i in 0..N { power[i] = power_msb(delta[i],4); }
    let mut term = [0; N];
    for i in 0..N { term[i] = floor_div_const::<4>(power[i]); }
    for i in 0..N { result[i] -= term[i]; }
    // (X - 1) ** 5 / 5
    let mut delta = [0; N];
    for i in 0..N { delta[i] = X[i]-SCALE; }
    let mut power = [0; N];
    for i in 0..N { power[i] = power_msb(delta[i],5); }
    let mut term = [0; N];
    for i in 0..N { term[i] = floor_div_const::<5>(power[i]); }
    for i in 0..N { result[i] += term[i]; }
    // (X - 1) ** 6 / 6
    let mut delta = [0; N];
    for i in 0..N { delta[i] = X[i]-SCALE; }
    let mut power = [0; N];
    for i in 0..N { power[i] = power_msb(delta[i],6); }
    let mut term = [0; N];
    for i in 0..N { term[i] = floor_div_const::<6>(power[i]); }
    for i in 0..N { result[i] -= term[i]; }
    // (X - 1) ** 7 / 7
    let mut delta = [0; N];
    for i in 0..N { delta[i] = X[i]-SCALE; }
    let mut power = [0; N];
    for i in 0..N { power[i] = power_msb(delta[i],7); }
    let mut term = [0; N];
    for i in 0..N { term[i] = floor_div_const::<7>(power[i]); }
    for i in 0..N { result[i] += term[i]; }
    // (X - 1) ** 8 / 8
    let mut delta = [0; N];
    for i in 0..N { delta[i] = X[i]-SCALE; }
    let mut power = [0; N];
    for i in 0..N { power[i] = power_msb(delta[i],8); }
    let mut term = [0; N];
    for i in 0..N { term[i] = floor_div_const::<8>(power[i]); }
    for i in 0..N { result[i] -= term[i]; }
    // (X - 1) ** 9 / 9
    let mut delta = [0; N];
    for i in 0..N { delta[i] = X[i]-SCALE; }
    let mut power = [0; N];
    for i in 0..N { power[i] = power_msb(delta[i],9); }
    let mut term = [0; N];
    for i in 0..N { term[i] = floor_div_const::<9>(power[i]); }
    for i in 0..N { result[i] += term[i]; }
    result
}
fn power_msb(x: Field,n: u32) -> Field {
    let mut value = x;
    let mut started = false;
    for t in 0..32 {
        let bit = (n >> (31-t)) & 1;
        if started { value=mul(value,value); if bit == 1 { value=mul(value,x); } }
        if bit == 1 { started=true; }
    }
    value
}
