// pythran_insertion_sort__insertion_sort (Pythran insertion_sort), Noir baseline.
// Kernel: insertion sort of `list2: Float[N]` in place; the output is list2
// after the call (`mutated_outputs`), i.e. np.sort(list2).
// The order is computed unconstrained and verified: element i has the key
// window(list2[i]) * 2^KEY_SHIFT + i, and the keys read through the claimed
// order must strictly increase (one range check of each step). The keys are
// distinct, so the order is a permutation, and the values read through it are
// sorted. KEY_SHIFT makes the step check 70 bits wide, five 14-bit limbs: no
// range table beyond the 64-bit checks' (NOIR_BASELINES.md).
// @outputs: float[N]

use noir_hw::{array_index, check_inputs, window};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = list2_0
global KEY_SHIFT: u32 = 6; // = 14 * cdiv(64 + clog2(list2_0), 14) - 64
global KEY_BITS: u32 = 64 + KEY_SHIFT;
// `array_index` width: 8 bits up to 128 elements, 28 above (NOIR_BASELINES.md).
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (list2_0 - 1) // 128)

// A stable ascending order of the windowed values (insertion sort).
unconstrained fn order_hint(list2: [Field; N]) -> [Field; N] {
    let mut w: [u64; N] = [0; N];
    let mut order: [u32; N] = [0; N];
    for i in 0..N {
        w[i] = window(list2[i]);
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

fn main(list2: [Field; N]) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(list2);
    // Safety: pinned by the key checks below.
    let order = unsafe { order_hint(list2) };
    let shift = 2.pow_32(KEY_SHIFT as Field);
    let mut sorted: [Field; N] = [0; N];
    let mut previous: Field = 0;
    for k in 0..N {
        // `order[k]` in [0, N - 1] (`array_index`), so the key's index part
        // fits below 2^KEY_SHIFT and the read is in bounds.
        let i = order[k];
        let v = list2[array_index::<INDEX_BITS>(i, (N - 1) as Field)];
        let key = (window(v) as Field) * shift + i;
        if k > 0 {
            (key - previous - 1).assert_max_bit_size::<KEY_BITS>();
        }
        previous = key;
        sorted[k] = v;
    }
    sorted
}
