// DS-1000 case 0358, Noir baseline.
// Kernel: `b[np.argsort(a.sum(axis=(1, 2))), :, :]` on a: Float[3, 3, 3], b: Integer[3, 3, 3].
// The three matrices of b in the ascending order of the sums of the matrices
// of a. NumPy sorts three elements by insertion, which is stable: ties keep
// their order. From the three comparisons s01 = s1 < s0, s02 = s2 < s0,
// s12 = s2 < s1, the one-hot selectors of the first matrix are
// (1 - f1 - f2, s01 (1 - s12), s02 s12) and of the last
// (s01 s02, (1 - s01) s12, 1 - l0 - l1); the middle selector is what remains.
// Each output cell is a selector's inner product with the three b cells.
// @outputs: int[27]

use noir_hw::window;

fn main(a: [Field; 27], b: [Field; 27]) -> pub [Field; 27] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    let mut s = [0; 3];
    for m in 0..3 {
        for p in 0..9 {
            s[m] += a[9 * m + p];
        }
    }
    let (w0, w1, w2) = (window(s[0]), window(s[1]), window(s[2]));
    let (s01, s02, s12) = ((w1 < w0) as Field, (w2 < w0) as Field, (w2 < w1) as Field);
    let f1 = s01 - s01 * s12;
    let f2 = s02 * s12;
    let l0 = s01 * s02;
    let l1 = s12 - s01 * s12;
    let first = [1 - f1 - f2, f1, f2];
    let last = [l0, l1, 1 - l0 - l1];
    let middle = [1 - first[0] - last[0], 1 - first[1] - last[1], 1 - first[2] - last[2]];
    let selectors = [first, middle, last];
    let mut result = [0; 27];
    for r in 0..3 {
        for p in 0..9 {
            let sel = selectors[r];
            result[9 * r + p] = sel[0] * b[p] + sel[1] * b[9 + p] + sel[2] * b[18 + p];
        }
    }
    result
}
