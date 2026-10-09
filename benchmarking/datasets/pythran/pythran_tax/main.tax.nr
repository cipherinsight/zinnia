// pythran_tax__tax (Pythran tax), Noir baseline.
// Kernel: progressive income tax of d: Float[N]: four masked selections by
// the constant thresholds 256303 > 54057 > 13769 > 8820, each mapped and
// summed, the total divided by np.sum(d).
// Only the sums are output, so each income contributes its own segment's term
// to one sum (a masked element of the other segments contributes 0). The four
// segment terms have one shape, s * (s * k + b) + e with s = d - t (segments 1
// and 2: t = 0, k = 0, b their rate), so every income evaluates it once, with
// its segment's constants selected by a one-hot of the segment: two rounded
// products per income, as NumPy rounds them in its own segment
// (floor(d * 0) = 0, exact). Below the lowest threshold every constant is 0
// and the term is 0. The segment one-hot is a hint pinned by two range
// checks (`segment`; 3,302 gates against 3,337 with four comparisons of the
// window offsets).
// @outputs: float[1]

use noir_hw::{check_inputs, div, mul, window};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = d_0

// The kernel's four tax segments (its text, not a size).
global SEGMENTS: u32 = 4;

// The thresholds as Q32 codes, highest first: segment s (1 to 4) holds the
// incomes in (THRESHOLDS[s - 1], THRESHOLDS[s - 2]] (no upper bound for
// segment 1); round(t * 2^32) of 256303, 54057, 13769, 8820.
global THRESHOLDS: [Field; SEGMENTS] = [1100813002866688, 232173047119872, 59137404698624, 37881611550720];
global THRESHOLDS_U64: [u64; SEGMENTS] = [
    9224472849857642496, 9223604209901895680, 9223431174259474432, 9223409918466326528,
]; // THRESHOLDS + 2^63, the window offsets
// Per segment 1 to 4, round(v * 2^32) of: the offset t subtracted from d
// (13769, 8820), the slope k (2.2376e-06, 1.00727e-05), the base b (0.45,
// 0.42, 0.2397, 0.14) and the constant e (-16164.53, -8475.44, 939.57) of the
// term s * (s * k + b) + e.
global OFFSET: [Field; SEGMENTS] = [0, 0, 59137404698624, 37881611550720];
global SLOPE: [Field; SEGMENTS] = [0, 0, 9610, 43262];
global BASE: [Field; SEGMENTS] = [1932735283, 1803886264, 1029503661, 601295421];
global CONSTANT: [Field; SEGMENTS] = [-69426127705211, -36401737619210, 4035422422303, 0];
// (lo, hi] of segment 0 (below the lowest threshold): lo = -2^63 - 1, so that
// every value in the window lies above it.
global FLOOR: Field = -9223372036854775809;
global CEILING: Field = 9223372036854775807; // 2^63 - 1, hi of segment 1

unconstrained fn segment_hint(x: Field) -> [Field; SEGMENTS] {
    let w = window(x);
    let mut hot: [Field; SEGMENTS] = [0; SEGMENTS];
    let mut found = false;
    for s in 0..SEGMENTS {
        if !found & (w > THRESHOLDS_U64[s]) {
            hot[s] = 1;
            found = true;
        }
    }
    hot
}

// The one-hot [h1, h2, h3, h4] of the segment of x (all zero below the lowest
// threshold).
fn segment(x: Field) -> [Field; SEGMENTS] {
    // With h_s bits summing to at most 1, the selected bounds (lo, hi] are one
    // segment's (the window's ends for the open sides), and lo < x <= hi holds
    // for exactly one segment, since they partition the window (x is in it).
    // Safety: pinned by the checks below.
    let hot = unsafe { segment_hint(x) };
    let mut any: Field = 0;
    let mut lo = FLOOR;
    let mut hi = THRESHOLDS[SEGMENTS - 1];
    for s in 0..SEGMENTS {
        assert(hot[s] * (hot[s] - 1) == 0, "segment: not a bit");
        any += hot[s];
        lo += hot[s] * (THRESHOLDS[s] - FLOOR);
        let upper = if s == 0 { CEILING } else { THRESHOLDS[s - 1] };
        hi += hot[s] * (upper - THRESHOLDS[SEGMENTS - 1]);
    }
    assert(any * (any - 1) == 0, "segment: more than one");
    (x - lo - 1).assert_max_bit_size::<64>();
    (hi - x).assert_max_bit_size::<64>();
    hot
}

fn pick(hot: [Field; SEGMENTS], values: [Field; SEGMENTS]) -> Field {
    let mut picked: Field = 0;
    for s in 0..SEGMENTS {
        picked += hot[s] * values[s];
    }
    picked
}

fn main(d: [Field; N]) -> pub Field {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(d);
    let mut total: Field = 0;
    let mut income: Field = 0;
    for i in 0..N {
        let hot = segment(d[i]);
        let s = d[i] - pick(hot, OFFSET);
        let p = mul(s, pick(hot, SLOPE)) + pick(hot, BASE);
        total += mul(s, p) + pick(hot, CONSTANT);
        income += d[i];
    }
    div(total, income)
}
