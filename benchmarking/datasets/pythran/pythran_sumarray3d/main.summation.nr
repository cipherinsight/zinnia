// pythran_sumarray3d__summation (Pythran sumarray3d), Noir baseline.
// Kernel: on pos: Float[NW, 3], weights: Float[NW], points: Float[NP, 3], for
// every point i and weight j, with (dx, dy, dz) = points[i] - pos[j]:
// dr = 1.0 / sqrt(dx * dx + dy * dy + dz * dz), total += weights[j] * dr and
// sum_array3d[i] += weights[j] * (dx, dy, dz); the output is
// (sum_array, sum_array3d), sum_array[i] the total of point i.
// Each square and weighted product rounds before addition, in the source
// recurrence. No sum-of-products reassociation is used.
// @outputs: float[NP]; float[NP * AXES]

use noir_hw::{check_inputs, divmod_pos, mul, sqrt, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global NW: u32 = 16; // = weights_0
global NP: u32 = 16; // = points_0
// The coordinates x, y, z: fixed by the kernel's text, not a size.
global AXES: u32 = 3;

fn main(pos: [[Field; AXES]; NW], weights: [Field; NW], points: [[Field; AXES]; NP]) -> pub ([Field; NP], [[Field; AXES]; NP]) {
    // Input domain: every input element is a Q32 code in the window.
    for j in 0..NW {
        check_inputs(pos[j]);
    }
    check_inputs(weights);
    for i in 0..NP {
        check_inputs(points[i]);
    }
    let mut sum_array: [Field; NP] = [0; NP];
    let mut sum_array3d: [[Field; AXES]; NP] = [[0; AXES]; NP];
    for i in 0..NP {
        let mut total: Field = 0;
        let mut sums: [Field; AXES] = [0; AXES];
        for j in 0..NW {
            let mut squares: Field = 0;
            for a in 0..AXES {
                let d = points[i][a] - pos[j][a];
                squares += mul(d, d);
                sums[a] += mul(weights[j], d);
            }
            // 1.0 / root: a quotient floor(2^64 / root), root > 0 (divmod_pos).
            let (dr, _) = divmod_pos(SCALE * SCALE, sqrt(squares));
            total += mul(weights[j], dr);
        }
        sum_array[i] = total;
        for a in 0..AXES {
            sum_array3d[i][a] = sums[a];
        }
    }
    (sum_array, sum_array3d)
}
