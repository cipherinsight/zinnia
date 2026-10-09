// DS-1000 case 0300: literal translation of kernel.py (LITERAL_BASELINES.md).
// Percentile: stable bubble sort (190 swaps), exact integer rank split,
// Interpolation rounds once: s_lo + floor(rem * (s_hi-s_lo) / 100), as optimized.
// p is checked in [0,100]; endpoint hi is clamped to the last index.
// @outputs: float[1]
fn main(a: [Field; 20], p: Field) -> pub Field {
    noir_hw::check_inputs(a);
    noir_hw::check_input(p);
    assert(noir_hw::ge(p, 0) & noir_hw::le(p, 100));
    let mut s = a;
    for pass in 0..19 {
        for j in 0..19-pass {
            let swap = noir_hw::gt(s[j], s[j+1]);
            let left = if swap { s[j+1] } else { s[j] };
            let right = if swap { s[j] } else { s[j+1] };
            s[j] = left;
            s[j+1] = right;
        }
    }
    let (lo, rem) = noir_hw::divmod_pos(19 * p, 100);
    let hi = noir_hw::min(lo + 1, 19);
    let s_lo = s[noir_hw::index_pos::<8>(lo, 20)];
    let s_hi = s[noir_hw::index_pos::<8>(hi, 20)];
    s_lo + noir_hw::divmod_pos(rem * (s_hi - s_lo), 100).0
}
