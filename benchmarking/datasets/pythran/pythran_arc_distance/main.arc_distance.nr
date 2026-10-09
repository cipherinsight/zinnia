// pythran_arc_distance__arc_distance (Pythran arc_distance), Noir baseline.
// Kernel: the haversine distance of points (theta_1, phi_1) and
// (theta_2, phi_2), each Float[N]:
// temp = sin((theta_2 - theta_1) / 2) ** 2
//        + cos(theta_1) * cos(theta_2) * sin((phi_2 - phi_1) / 2) ** 2,
// 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp)).
// v / 2 is floor(v / 2) (`floor_div_pow2`); sin, cos are the library's;
// every * and ** 2 is a rounded product, left to right; the square roots are
// exact floors, and as proven non-negative values they feed `arctan2_nonneg`
// (no sign handling). Where the rounded temp falls outside [0, 1] (possible
// only within a few units of 2^-32 of an end, as float64's temp can, which
// makes NumPy's result NaN) a square root has no witness.
// @outputs: float[N]

use noir_hw::{arctan2_nonneg, check_inputs, cos, floor_div_pow2, mul, sin, sqrt, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 5; // = theta_1_0

fn half_angle_sin_sq(from: Field, to: Field) -> Field {
    let s = sin(floor_div_pow2::<1>(to - from));
    mul(s, s)
}

fn main(theta_1: [Field; N], phi_1: [Field; N], theta_2: [Field; N], phi_2: [Field; N]) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(theta_1);
    check_inputs(phi_1);
    check_inputs(theta_2);
    check_inputs(phi_2);
    let mut result: [Field; N] = [0; N];
    for i in 0..N {
        let lat = half_angle_sin_sq(theta_1[i], theta_2[i]);
        let lon = half_angle_sin_sq(phi_1[i], phi_2[i]);
        let temp = lat + mul(mul(cos(theta_1[i]), cos(theta_2[i])), lon);
        let angle = arctan2_nonneg(sqrt(temp), sqrt(SCALE - temp));
        result[i] = 2 * angle;
    }
    result
}
