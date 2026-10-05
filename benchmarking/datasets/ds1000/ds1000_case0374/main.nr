// DS-1000 case 0374, Noir baseline.
// Kernel: ECDF lookup on grades: Float[50], eval: Integer[5]: with
// xs = sort(grades) and ys = arange(1, 51) / 50, each e of eval maps to 0 if
// e < xs[0], 1 if e >= xs[-1], else ys[(xs > e).argmax() - 1].
// For sorted xs, (xs > e).argmax() is the number c of grades <= e whenever
// some grade exceeds e, so every branch equals c / 50 (c = 0 below xs[0],
// c = 50 from xs[-1] on), ties included. The circuit counts c directly (no
// sort) and returns floor(c * 2^32 / 50), the Q32 value of ys[c - 1] = c / 50.
// e is compared as the float e * 2^32, which must lie in the window
// (-2^31 <= e < 2^31).
// @outputs: float[5]

use noir_hw::{divmod_pos, from_int, window, SCALE};

fn main(grades: [Field; 50], eval: [Field; 5]) -> pub [Field; 5] {
    let mut w: [u64; 50] = [0; 50];
    for i in 0..50 {
        w[i] = window(grades[i]);
    }
    let mut result: [Field; 5] = [0; 5];
    for k in 0..5 {
        let e = window(from_int(eval[k]));
        let mut c: Field = 0;
        for i in 0..50 {
            c += (w[i] <= e) as Field;
        }
        result[k] = divmod_pos(c * SCALE, 50).0;
    }
    result
}
