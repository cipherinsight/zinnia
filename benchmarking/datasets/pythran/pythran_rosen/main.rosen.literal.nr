// pythran_rosen__rosen: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize each elementwise array before the next operation, then sum in index order.
// @outputs: float[1]

use noir_hw::{check_inputs, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0

fn main(x: [Field; N]) -> pub Field {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(x);
    let mut squares = [0; N-1];
    for i in 0..N-1 { squares[i] = mul(x[i],x[i]); }
    let mut gaps = [0; N-1];
    for i in 0..N-1 { gaps[i] = x[i+1]-squares[i]; }
    let mut squared_gaps = [0; N-1];
    for i in 0..N-1 { squared_gaps[i] = mul(gaps[i],gaps[i]); }
    let mut t0 = [0; N-1];
    for i in 0..N-1 { t0[i] = 100*squared_gaps[i]; }
    let mut rest = [0; N-1];
    for i in 0..N-1 { rest[i] = SCALE-x[i]; }
    let mut t1 = [0; N-1];
    for i in 0..N-1 { t1[i] = mul(rest[i],rest[i]); }
    let mut sum = [0; N-1];
    for i in 0..N-1 { sum[i] = t0[i]+t1[i]; }
    let mut total: Field = 0;
    for i in 0..N-1 { total += sum[i]; }
    total
}
