// DS-1000 case 0375, Noir baseline.
// Kernel: on grades: Float[50] and threshold: Float, with xs = sort(grades)
// and ys = arange(1, 51) / 50: t = (ys > threshold).argmax(), return
// [xs[0], xs[t]].
// ys is constant, so t depends on threshold alone. The Q32 constants are
// Y_k = round(k / 50 * 2^32) = floor((k * 2^32 + 25) / 50) (k = 1..50), and for
// the Q32 threshold T, Y_k <= T iff k * 2^32 <= 50 T + 24, i.e.
// k <= u = floor((50 T + 24) / 2^32). So clamp(u, 0, 50) values of ys are
// <= threshold, and argmax (the first True, 0 when there is none) is t = u if
// 0 <= u < 50, else 0.
// xs[0] is the minimum of the grades (min_of). xs[t] is an
// order statistic at a private position: a hint v pinned by counting,
// #{g < v} <= t < #{g <= v}: 50 comparisons and 50 equality tests, where a
// sort would need a permutation argument on top of 49 comparisons.
// @outputs: float[2]

use noir_hw::{min_of, rescale, window};

global OFFSET_U64: u64 = 9223372036854775808; // 2^63

// The value at position k of xs sorted ascending: the x with
// #{y < x} <= k < #{y <= x}.
unconstrained fn kth_hint(xs: [Field; 50], k: u32) -> Field {
    let mut w: [u64; 50] = [0; 50];
    for i in 0..50 {
        w[i] = window(xs[i]);
    }
    let mut v: Field = 0;
    for i in 0..50 {
        let mut below: u32 = 0;
        let mut at_most: u32 = 0;
        for j in 0..50 {
            below += (w[j] < w[i]) as u32;
            at_most += (w[j] <= w[i]) as u32;
        }
        if (below <= k) & (k < at_most) {
            v = xs[i];
        }
    }
    v
}

fn main(grades: [Field; 50], threshold: Field) -> pub [Field; 2] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(grades);
    noir_hw::check_input(threshold);
    let low = min_of(grades);

    let u = rescale(50 * threshold + 24);
    let wu = window(u);
    let in_range = (wu >= OFFSET_U64) & (wu < OFFSET_U64 + 50);
    let t_field = if in_range { u } else { 0 };
    t_field.assert_max_bit_size::<6>(); // t in [0, 50), so the cast is exact
    let t = t_field as u32;

    // Safety: v is pinned by #{g < v} <= t < #{g <= v}, which at most one v
    // satisfies (for v < v' both would give #{g <= v} <= #{g < v'} <= t <
    // #{g <= v}); that v is the sorted value at position t, one of the grades.
    let v = unsafe { kth_hint(grades, t) };
    let wv = window(v);
    let mut below: u32 = 0;
    let mut equal: u32 = 0;
    for i in 0..50 {
        below += (window(grades[i]) < wv) as u32;
        equal += (grades[i] == v) as u32;
    }
    assert(below <= t);
    assert(t < below + equal);

    [low, v]
}
