// pythran_tax__tax (Pythran tax), Noir baseline.
// Kernel: progressive income tax of d: Float[N]: four masked selections by
// the constant thresholds 256303 > 54057 > 13769 > 8820, each mapped and
// summed, the total divided by np.sum(d).
// Preserve the source's four masks, compacted arrays and separate formulas.
// Each scalar product rounds at the source boundary; padding contributes zero.
// @outputs: float[1]

use noir_hw::{check_inputs, canonical_div_or_zero, mul, gt, select_kept};

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
    let mut totals: [Field; SEGMENTS] = [0; SEGMENTS];
    for segment in 0..SEGMENTS {
        let mut keep = [false; N];
        for i in 0..N {
            keep[i] = gt(d[i], THRESHOLDS[segment]) &
                (if segment == 0 { true } else { !gt(d[i], THRESHOLDS[segment - 1]) });
        }
        let selected = select_kept(d, keep);
        for i in 0..N {
            let active = gt(selected[0], i as Field);
            let x = selected[i + 1];
            let shifted = x - (if active { OFFSET[segment] } else { 0 });
            let product = if segment < 2 {
                mul(shifted, BASE[segment])
            } else {
                mul(shifted, mul(shifted, SLOPE[segment]) + BASE[segment])
            };
            totals[segment] += product + (if active { CONSTANT[segment] } else { 0 });
        }
    }
    let mut income: Field = 0;
    for x in d { income += x; }
    assert(income != 0, "source division by zero");
    canonical_div_or_zero(totals[0] + totals[1] + totals[2] + totals[3], income)
}
