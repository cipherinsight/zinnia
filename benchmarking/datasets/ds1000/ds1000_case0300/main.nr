// DS-1000 case 0300, Noir baseline.
// Kernel: `np.percentile(a, p)` on Float[20] with an Integer p.
// NumPy rejects p outside [0, 100] (ValueError: no proof). Default method
// 'linear': with 19 p = 100 lo + r (exact integers, lo in [0, 19]) the result
// is s[lo] + (r / 100) (s[hi] - s[lo]) for s = sorted(a), hi = min(lo + 1, 19);
// the product is floor(r (s[hi] - s[lo]) / 100), within one Q32 unit of the
// exact value.
// The sorted order is computed unconstrained and verified: element i has the
// key window(a_i) * 64 + i, and the keys read through the claimed order must
// strictly increase (`sorted`). s[lo] and s[hi] are native reads.
// @outputs: float[1]

use noir_hw::{divmod_pos, window};

global N: u32 = 20;

// The ascending argsort of the windowed values (insertion sort).
unconstrained fn argsort_hint(a: [Field; N]) -> [Field; N] {
    let mut w: [u64; N] = [0; N];
    let mut order: [u32; N] = [0; N];
    for i in 0..N {
        w[i] = window(a[i]);
        order[i] = i;
    }
    for i in 1..N {
        let mut j = i;
        while j > 0 {
            if w[order[j]] >= w[order[j - 1]] {
                break;
            }
            let t = order[j];
            order[j] = order[j - 1];
            order[j - 1] = t;
            j -= 1;
        }
    }
    let mut out: [Field; N] = [0; N];
    for k in 0..N {
        out[k] = order[k] as Field;
    }
    out
}

// (19 p) // 100 and (19 p) % 100 for p in [0, 100].
unconstrained fn rank_hint(p: Field) -> (Field, Field) {
    let g = 19 * (p as u32);
    ((g / 100) as Field, (g % 100) as Field)
}

fn main(a: [Field; N], p: Field) -> pub Field {
    // Safety: pinned by `sorted`.
    let order = unsafe { argsort_hint(a) };
    let s = sorted(a, order);
    // Safety: pinned by `check_rank`.
    let (lo, r) = unsafe { rank_hint(p) };
    check_rank(p, lo, r);
    let lo = lo as u32;
    let hi = if lo == N - 1 { lo } else { lo + 1 };
    let s_lo = s[lo];
    s_lo + divmod_pos(r * (s[hi] - s_lo), 100).0
}

// a read through `order`, checked to be sorted: each index is below 20
// (8-bit check, then the array read) and the keys window(a_i) * 64 + i read
// through the order strictly increase (70-bit checks: a multiple of the
// 14-bit limb, so no extra range table). A key is a function of its index, so
// the indices are distinct, a permutation of 0..20, and the one that sorts
// the values.
fn sorted(a: [Field; N], order: [Field; N]) -> [Field; N] {
    let mut s: [Field; N] = [0; N];
    let mut prev: Field = 0;
    for k in 0..N {
        let i = order[k];
        i.assert_max_bit_size::<8>();
        let v = a[i as u32];
        let key = (window(v) as Field) * 64 + i;
        if k > 0 {
            (key - prev - 1).assert_max_bit_size::<70>();
        }
        prev = key;
        s[k] = v;
    }
    s
}

// p in [0, 100] (NumPy's range) and 19 p = 100 lo + r with r in [0, 100) and
// lo in [0, 256): the unique Euclidean split.
fn check_rank(p: Field, lo: Field, r: Field) {
    p.assert_max_bit_size::<8>();
    (100 - p).assert_max_bit_size::<8>();
    r.assert_max_bit_size::<8>();
    (99 - r).assert_max_bit_size::<8>();
    lo.assert_max_bit_size::<8>();
    assert(19 * p == 100 * lo + r);
}
