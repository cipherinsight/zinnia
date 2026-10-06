// DS-1000 case 0381, Noir baseline.
// Kernel: `np.argsort(a)[::-1][:len(a)]` on Float[100]: the ascending argsort
// reversed (`[:len(a)]` keeps all 100).
// The order is computed unconstrained and verified: element i has the key
// window(a_i) * 128 + i, and the keys read through the claimed order must
// strictly increase (one 71-bit range check per step). The keys are distinct
// and ordered by (a_i, i), so this is the stable order (NumPy's
// kind="stable"; the default introsort leaves the order of equal values
// unspecified); reversed, equal values come out in decreasing index order.
// @outputs: int[100]

use noir_hw::{array_index, window};

global N: u32 = 100;

// The stable ascending argsort of the windowed values (insertion sort).
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

fn main(a: [Field; N]) -> pub [Field; N] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    // Safety: pinned by `check_order`.
    let order = unsafe { argsort_hint(a) };
    check_order(a, order);
    let mut reversed: [Field; N] = [0; N];
    for k in 0..N {
        reversed[k] = order[N - 1 - k];
    }
    reversed
}

// `order` is the stable ascending argsort of a: each index is in [0, 99]
// (noir_hw::array_index, whose checks the compiler keeps, then the array
// read) and the keys window(a_i) * 128 + i read through the order strictly
// increase (71-bit check of each step). A key is a function of its index, so
// the indices are distinct, a permutation of 0..100, and the one that sorts
// the keys.
fn check_order(a: [Field; N], order: [Field; N]) {
    let mut prev: Field = 0;
    for k in 0..N {
        let i = order[k];
        let v = a[array_index::<8>(i, (N - 1) as Field)];
        let key = (window(v) as Field) * 128 + i;
        if k > 0 {
            (key - prev - 1).assert_max_bit_size::<71>();
        }
        prev = key;
    }
}
