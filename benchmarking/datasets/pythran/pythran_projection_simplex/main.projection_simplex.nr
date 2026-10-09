// pythran_projection_simplex__projection_simplex (Pythran projection_simplex), Noir baseline.
// Kernel: Euclidean projection of `v: Float[N]` onto the simplex of radius z
// (an integer): u = sort(v) descending, cssv = cumsum(u) - z,
// cond[k] = u[k] - cssv[k] / (k + 1) > 0, rho the last k + 1 with cond[k] (no
// such k raises IndexError: no witness), theta = cssv[rho - 1] / rho, output
// max(v - theta, 0).
// The ascending order is computed unconstrained and verified as in the
// insertion_sort pilot (the keys window(v[i]) * 2^KEY_SHIFT + i read through
// it strictly increase), then read backwards. cssv[k] / (k + 1) divides by an
// integer constant (`divmod_pos`; by 1 it is cssv[0] itself), and theta is the
// same quotient at k = rho - 1, so it is selected from them by the one-hot of
// the last true condition (cond[k] * prod_{m > k} (1 - cond[m])), whose sum
// must be 1.
// @outputs: float[N]

use noir_hw::{array_index, check_input, check_inputs, divmod_pos, window, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = v_0
global KEY_SHIFT: u32 = 6; // = 14 * cdiv(64 + clog2(v_0), 14) - 64
global KEY_BITS: u32 = 64 + KEY_SHIFT;
// `array_index` width: 8 bits up to 128 elements, 28 above (NOIR_BASELINES.md).
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (v_0 - 1) // 128)

// A stable ascending order of the windowed values (insertion sort).
unconstrained fn order_hint(v: [Field; N]) -> [Field; N] {
    let mut w: [u64; N] = [0; N];
    let mut order: [u32; N] = [0; N];
    for i in 0..N {
        w[i] = window(v[i]);
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

fn main(v: [Field; N], z: Field) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code or an int64 in the window.
    check_inputs(v);
    check_input(z);
    // u = np.sort(v)[::-1], with the windows of its values.
    // Safety: pinned by the key checks below.
    let order = unsafe { order_hint(v) };
    let shift = 2.pow_32(KEY_SHIFT as Field);
    let mut u: [Field; N] = [0; N];
    let mut u_window: [u64; N] = [0; N];
    let mut previous: Field = 0;
    for k in 0..N {
        // `order[k]` in [0, N - 1] (`array_index`), so the key's index part
        // fits below 2^KEY_SHIFT and the read is in bounds.
        let i = order[k];
        let value = v[array_index::<INDEX_BITS>(i, (N - 1) as Field)];
        let w = window(value);
        let key = (w as Field) * shift + i;
        if k > 0 {
            (key - previous - 1).assert_max_bit_size::<KEY_BITS>();
        }
        previous = key;
        u[N - 1 - k] = value;
        u_window[N - 1 - k] = w;
    }
    // cssv[k] / (k + 1) and cond[k]
    let z_float = z * SCALE;
    let mut cumsum: Field = 0;
    let mut quotients: [Field; N] = [0; N];
    let mut cond: [Field; N] = [0; N];
    for k in 0..N {
        cumsum += u[k];
        let cssv = cumsum - z_float;
        let quotient = if k == 0 {
            cssv
        } else {
            let (quotient, _) = divmod_pos(cssv, (k + 1) as Field);
            quotient
        };
        quotients[k] = quotient;
        cond[k] = (window(quotient) < u_window[k]) as Field;
    }
    // theta: the quotient at the last k with cond[k].
    let mut none_after: Field = 1;
    let mut hits: Field = 0;
    let mut theta: Field = 0;
    for t in 0..N {
        let k = N - 1 - t;
        let last = cond[k] * none_after;
        hits += last;
        theta += last * quotients[k];
        none_after *= 1 - cond[k];
    }
    assert(hits == 1, "no index meets the condition (IndexError)");
    // np.maximum(v - theta, 0)
    let mut w: [Field; N] = [0; N];
    for i in 0..N {
        let gap = v[i] - theta;
        w[i] = if window(gap) > window(0) { gap } else { 0 };
    }
    w
}
