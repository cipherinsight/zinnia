// DS-1000 case 0420, Noir baseline.
// Kernel: `np.where(x < x_min, x_min, np.where(x > x_max, x_max, 3*x**2 - 2*x**3))`
// with x: Float, x_min, x_max: Integer (a float result).
// The bounds are compared as floats (float(x_min), float(x_max), which like
// every float must lie in the Q32 window). The polynomial is evaluated at x
// when x_min <= x <= x_max and at 0 otherwise: its value is only selected in
// the first case, and a clamped x far outside the bounds would otherwise
// overflow x**3 although NumPy's result (a bound) fits.
// @outputs: float[1]

use noir_hw::{from_int, mul, window};

fn main(x: Field, x_min: Field, x_max: Field) -> pub Field {
    let lo = from_int(x_min);
    let hi = from_int(x_max);
    let wx = window(x);
    let below = wx < window(lo);
    let above = wx > window(hi);
    let t = if below | above { 0 } else { x };
    let t2 = mul(t, t);
    let t3 = mul(t2, t);
    if below {
        lo
    } else if above {
        hi
    } else {
        3 * t2 - 2 * t3
    }
}
