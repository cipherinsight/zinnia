// DS-1000 case 0357, Noir baseline.
// Kernel: `c = b[np.argsort(a, axis=0)[::-1], j, k]` on a: Float[3, 3, 3], b: Integer[3, 3, 3].
// Column c[:, j, k] is b[:, j, k] in the reverse of the ascending order of
// a[:, j, k]. NumPy sorts three elements by insertion, which is stable: ties
// keep their order before the reversal. From the three comparisons
// s01 = a1 < a0, s02 = a2 < a0, s12 = a2 < a1, the first element of the
// ascending order is s12 ? (s02 ? 2 : 0) : (s01 ? 1 : 0) and the last
// s12 ? (s01 ? 0 : 1) : (s02 ? 0 : 2); their b values are selected directly
// (no index gather; `x + s (y - x)` is `s ? y : x`), and the middle one is the
// column sum minus those two.
// @outputs: int[27]

use noir_hw::window;

fn main(a: [Field; 27], b: [Field; 27]) -> pub [Field; 27] {
    let mut c = [0; 27];
    for col in 0..9 {
        let (a0, a1, a2) = (window(a[col]), window(a[9 + col]), window(a[18 + col]));
        let (b0, b1, b2) = (b[col], b[9 + col], b[18 + col]);
        let (s01, s02, s12) = ((a1 < a0) as Field, (a2 < a0) as Field, (a2 < a1) as Field);
        let first_if = b0 + s02 * (b2 - b0);
        let first_else = b0 + s01 * (b1 - b0);
        let first = first_else + s12 * (first_if - first_else);
        let last_if = b1 + s01 * (b0 - b1);
        let last_else = b2 + s02 * (b0 - b2);
        let last = last_else + s12 * (last_if - last_else);
        c[col] = last;
        c[9 + col] = b0 + b1 + b2 - first - last;
        c[18 + col] = first;
    }
    c
}
