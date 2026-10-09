// npbench_pythran_arc_distance_arc_distance__arc_distance: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each array operation completes in expression order; shared canonical transcendental arithmetic.

// @outputs: float[N]

use noir_hw::{canonical_arctan2 as arctan2, check_inputs, canonical_cos as cos, floor_div_pow2, mul, canonical_sin as sin, sqrt, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = theta_1_0

fn main(theta_1: [Field; N], phi_1: [Field; N], theta_2: [Field; N], phi_2: [Field; N]) -> pub [Field; N] {
    check_inputs(theta_1); check_inputs(phi_1); check_inputs(theta_2); check_inputs(phi_2);
    let mut dt=[0;N]; for i in 0..N { dt[i]=theta_2[i]-theta_1[i]; }
    let mut ht=[0;N]; for i in 0..N { ht[i]=floor_div_pow2::<1>(dt[i]); }
    let mut st=[0;N]; for i in 0..N { st[i]=sin(ht[i]); }
    let mut lat=[0;N]; for i in 0..N { lat[i]=mul(st[i],st[i]); }
    let mut c1=[0;N]; for i in 0..N { c1[i]=cos(theta_1[i]); }
    let mut c2=[0;N]; for i in 0..N { c2[i]=cos(theta_2[i]); }
    let mut cc=[0;N]; for i in 0..N { cc[i]=mul(c1[i],c2[i]); }
    let mut dp=[0;N]; for i in 0..N { dp[i]=phi_2[i]-phi_1[i]; }
    let mut hp=[0;N]; for i in 0..N { hp[i]=floor_div_pow2::<1>(dp[i]); }
    let mut sp=[0;N]; for i in 0..N { sp[i]=sin(hp[i]); }
    let mut lon=[0;N]; for i in 0..N { lon[i]=mul(sp[i],sp[i]); }
    let mut weighted=[0;N]; for i in 0..N { weighted[i]=mul(cc[i],lon[i]); }
    let mut temp=[0;N]; for i in 0..N { temp[i]=lat[i]+weighted[i]; }
    let mut y=[0;N]; for i in 0..N { y[i]=sqrt(temp[i]); }
    let mut rest=[0;N]; for i in 0..N { rest[i]=SCALE-temp[i]; }
    let mut x=[0;N]; for i in 0..N { x[i]=sqrt(rest[i]); }
    let mut angles=[0;N]; for i in 0..N { angles[i]=arctan2(y[i],x[i]); }
    let mut result=[0;N]; for i in 0..N { result[i]=2*angles[i]; }
    result
}
