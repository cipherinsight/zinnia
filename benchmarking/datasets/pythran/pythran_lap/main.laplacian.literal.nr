// pythran_lap__laplacian: literal translation of kernel.py (LITERAL_BASELINES.md).
// Nine slice statements are evaluated separately, retaining lap[0]=4/3*var[1]. Fresh division gadget per output.
// @outputs: float[N]

use noir_hw::{check_input, check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = var_0

global NEAR: Field = 5726623061; // round(4/3 * 2^32)
global CENTER: Field = -10737418240; // round(-5/2 * 2^32)
global FAR: Field = -357913941; // round(-1/12 * 2^32)



fn main(var: [Field; N], dh2: Field) -> pub [Field; N] {
    check_inputs(var); check_input(dh2);
    assert(dh2 != 0);
    let mut lap = [0; N];
    let mut products0 = [0; N];
    for i in 1..N { products0[i] = mul(NEAR, var[i-1]); }
    for i in 1..N { lap[i] = products0[i]; }
    let mut products1 = [0; N];
    for i in 0..1 { products1[i] = mul(NEAR, var[1]); }
    for i in 0..1 { lap[i] = products1[i]; }
    let mut products2 = [0; N];
    for i in 0..N-1 { products2[i] = mul(NEAR, var[i+1]); }
    for i in 0..N-1 { lap[i] += products2[i]; }
    let mut products3 = [0; N];
    for i in N-1..N { products3[i] = mul(NEAR, var[0]); }
    for i in N-1..N { lap[i] += products3[i]; }
    let mut products4 = [0; N];
    for i in 0..N { products4[i] = mul(CENTER, var[i]); }
    for i in 0..N { lap[i] += products4[i]; }
    let mut products5 = [0; N];
    for i in 2..N { products5[i] = mul(FAR, var[i-2]); }
    for i in 2..N { lap[i] += products5[i]; }
    let mut products6 = [0; N];
    for i in 0..2 { products6[i] = mul(FAR, var[N-2+i]); }
    for i in 0..2 { lap[i] += products6[i]; }
    let mut products7 = [0; N];
    for i in 0..N-2 { products7[i] = mul(FAR, var[i+2]); }
    for i in 0..N-2 { lap[i] += products7[i]; }
    let mut products8 = [0; N];
    for i in N-2..N { products8[i] = mul(FAR, var[i-(N-2)]); }
    for i in N-2..N { lap[i] += products8[i]; }
    let mut result = [0; N];
    for i in 0..N { result[i] = noir_hw::div(lap[i],dh2); }
    result
}
