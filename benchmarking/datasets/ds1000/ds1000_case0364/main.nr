// DS-1000 case 0364, Noir baseline.
// Kernel: `np.insert(a, pos, element, axis=0)` on Float[100, 10] with an
// Integer pos and a Float[10] row.
// NumPy accepts pos in [-100, 100] (negative counts from the end) and raises
// IndexError otherwise (no proof). With P the insertion point and
// g_k = [P < k], output row k is a[k] (k < P), element (k = P) or a[k-1]
// (k > P), i.e. a[k] + h[k] - h[k-1] per cell with h[k] = g_{k+1} (e - a[k]):
// one product per output cell. The thermometer g is a hint pinned by
// `insertion_point` with one-bit checks only.
// @outputs: float[1010]

global N: u32 = 100;
global W: u32 = 10;

// The sign of pos and the thermometer g[k - 1] = [P < k], k = 1..=N.
unconstrained fn insertion_hint(pos: Field) -> (bool, [bool; N]) {
    let neg = (pos as u128) >= 18446744073709551616; // pos < 0
    let p = if neg { (pos + N as Field) as u32 } else { pos as u32 };
    let mut g: [bool; N] = [false; N];
    for k in 1..N + 1 {
        g[k - 1] = p < k;
    }
    (neg, g)
}

// The insertion point P of np.insert on an axis of length N (P = pos for pos
// in [0, N], pos + N for pos in [-N, -1]) as g[k - 1] = [P < k].
fn insertion_point(pos: Field) -> [bool; N] {
    // Safety: pinned by `check_insertion_point`.
    let (neg, g) = unsafe { insertion_hint(pos) };
    check_insertion_point(pos, neg, g);
    g
}

// The bools are one-bit checked by the compiler; g is monotone (g[k] implies
// g[k + 1]), so it is the thermometer of its number of zeros Z; Z = pos + N neg;
// and neg implies g[N - 1] (Z < N). For pos in [0, N] only neg = false fits,
// for pos in [-N, -1] only neg = true; any other pos gives Z outside [0, N]:
// no witness.
fn check_insertion_point(pos: Field, neg: bool, g: [bool; N]) {
    let mut zeros: Field = 0;
    for k in 0..N {
        if k + 1 < N {
            assert(!g[k] | g[k + 1]);
        }
        zeros += (!g[k]) as Field;
    }
    assert(zeros == pos + if neg { N as Field } else { 0 }, "index out of bounds");
    assert(!neg | g[N - 1]);
}

fn main(a: [Field; N * W], pos: Field, element: [Field; W]) -> pub [Field; (N + 1) * W] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_input(pos);
    noir_hw::check_inputs(element);
    let g = insertion_point(pos);
    // out[k] = a[k] + h[k] - h[k-1] with h[k] = g_{k+1} (e - a[k]): k < P gives
    // a[k], k = P gives e, k > P gives a[k-1] (g_0 = 0, g_{N+1} = 1).
    let mut out: [Field; (N + 1) * W] = [0; (N + 1) * W];
    let mut h_prev: [Field; W] = [0; W];
    for k in 0..N + 1 {
        for c in 0..W {
            let h = if k < N { (g[k] as Field) * (element[c] - a[k * W + c]) } else { element[c] };
            std::as_witness(h); // one witness, reused by output k + 1
            let cur = if k < N { a[k * W + c] } else { 0 };
            out[k * W + c] = cur + h - h_prev[c];
            h_prev[c] = h;
        }
    }
    out
}
