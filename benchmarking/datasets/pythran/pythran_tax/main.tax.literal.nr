// pythran_tax__tax: literal translation of kernel.py (LITERAL_BASELINES.md).
// CHECKER BLOCKER: Halo2 L5/L6 per-operation input bounds conflict with the checker's
// zero redundant_input_checks requirement; see LITERAL_NPB3.md. No diagnostic bypass.
// Each mask compacts by running count and equality tests per input/output pair; padding effects are gated after array operations.
// @outputs: float[1]

use noir_hw::{check_inputs, canonical_div_nonzero, mul, gt};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = d_0

// The kernel's four tax segments (its text, not a size).
global SEGMENTS: u32 = 4;

// The thresholds as Q32 codes, highest first: segment s (1 to 4) holds the
// incomes in (THRESHOLDS[s - 1], THRESHOLDS[s - 2]] (no upper bound for
// segment 1); round(t * 2^32) of 256303, 54057, 13769, 8820.
global THRESHOLDS: [Field; SEGMENTS] = [1100813002866688, 232173047119872, 59137404698624, 37881611550720];
// Per segment 1 to 4, round(v * 2^32) of: the offset t subtracted from d
// (13769, 8820), the slope k (2.2376e-06, 1.00727e-05), the base b (0.45,
// 0.42, 0.2397, 0.14) and the constant e (-16164.53, -8475.44, 939.57) of the
// term s * (s * k + b) + e.
global OFFSET: [Field; SEGMENTS] = [0, 0, 59137404698624, 37881611550720];
global SLOPE: [Field; SEGMENTS] = [0, 0, 9610, 43262];
global BASE: [Field; SEGMENTS] = [1932735283, 1803886264, 1029503661, 601295421];
global CONSTANT: [Field; SEGMENTS] = [-69426127705211, -36401737619210, 4035422422303, 0];
fn main(d: [Field; N]) -> pub Field {
    check_inputs(d);
    // Source mask and compacted segment 1.
    let mut lower=[false; N]; for i in 0..N { lower[i]=gt(d[i],THRESHOLDS[0]); }
    let keep=lower;
    let (s0,active0)=compact_scan(d,keep);
    let mut product=[0; N]; for p in 0..N { product[p]=mul(s0[p],BASE[0]); }
    let mut seg0=[0; N]; for p in 0..N { seg0[p]=(product[p]+CONSTANT[0])*active0[p]; }
    // Source mask and compacted segment 2.
    let mut lower=[false; N]; for i in 0..N { lower[i]=gt(d[i],THRESHOLDS[1]); }
    let mut upper=[false; N]; for i in 0..N { upper[i]=noir_hw::le(d[i],THRESHOLDS[0]); }
    let mut keep=[false; N]; for i in 0..N { keep[i]=lower[i] & upper[i]; }
    let (s1,active1)=compact_scan(d,keep);
    let mut product=[0; N]; for p in 0..N { product[p]=mul(s1[p],BASE[1]); }
    let mut seg1=[0; N]; for p in 0..N { seg1[p]=(product[p]+CONSTANT[1])*active1[p]; }
    // Source mask and compacted segment 3.
    let mut lower=[false; N]; for i in 0..N { lower[i]=gt(d[i],THRESHOLDS[2]); }
    let mut upper=[false; N]; for i in 0..N { upper[i]=noir_hw::le(d[i],THRESHOLDS[1]); }
    let mut keep=[false; N]; for i in 0..N { keep[i]=lower[i] & upper[i]; }
    let (s2,active2)=compact_scan(d,keep);
    let mut seg2=[0; N]; for p in 0..N { seg2[p]=(s2[p]-OFFSET[2])*active2[p]; }
    // Source mask and compacted segment 4.
    let mut lower=[false; N]; for i in 0..N { lower[i]=gt(d[i],THRESHOLDS[3]); }
    let mut upper=[false; N]; for i in 0..N { upper[i]=noir_hw::le(d[i],THRESHOLDS[2]); }
    let mut keep=[false; N]; for i in 0..N { keep[i]=lower[i] & upper[i]; }
    let (s3,active3)=compact_scan(d,keep);
    let mut seg3=[0; N]; for p in 0..N { seg3[p]=(s3[p]-OFFSET[3])*active3[p]; }
    let mut product=[0; N]; for p in 0..N { product[p]=mul(seg2[p],SLOPE[2]); }
    let mut prog2=[0; N]; for p in 0..N { prog2[p]=(product[p]+BASE[2])*active2[p]; }
    let mut product=[0; N]; for p in 0..N { product[p]=mul(seg3[p],SLOPE[3]); }
    let mut prog3=[0; N]; for p in 0..N { prog3[p]=(product[p]+BASE[3])*active3[p]; }
    let terms=seg0;
    let mut total0: Field=0; for p in 0..N { total0+=terms[p]; }
    let terms=seg1;
    let mut total1: Field=0; for p in 0..N { total1+=terms[p]; }
    let subtotal1=total0+total1;
    let mut terms=[0; N]; for p in 0..N { terms[p]=mul(seg2[p],prog2[p]); }
    for p in 0..N { terms[p]=(terms[p]+CONSTANT[2])*active2[p]; }
    let mut total2: Field=0; for p in 0..N { total2+=terms[p]; }
    let subtotal2=subtotal1+total2;
    let mut terms=[0; N]; for p in 0..N { terms[p]=mul(seg3[p],prog3[p]); }
    let mut total3: Field=0; for p in 0..N { total3+=terms[p]; }
    let subtotal3=subtotal2+total3;
    let mut income: Field=0; for p in 0..N { income+=d[p]; }
    canonical_div_nonzero(subtotal3,income)
}
fn compact_scan(x: [Field; N],keep: [bool; N]) -> ([Field; N],[Field; N]) {
    let mut count: Field=0; let mut positions=[0; N];
    for i in 0..N { positions[i]=count; count+=keep[i] as Field; }
    let mut selected=[0; N]; let mut active=[0; N];
    for j in 0..N {
        for i in 0..N {
            let hit=(keep[i] & (positions[i] == j as Field)) as Field;
            selected[j]+=hit*x[i]; active[j]+=hit;
        }
    }
    (selected,active)
}
