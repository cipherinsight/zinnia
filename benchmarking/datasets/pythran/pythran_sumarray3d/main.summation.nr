// pythran_sumarray3d__summation (Pythran sumarray3d), Noir baseline.
// Kernel: on pos: Float[NW, 3], weights: Float[NW], points: Float[NP, 3], for
// every point i and weight j, with (dx, dy, dz) = points[i] - pos[j]:
// dr = 1.0 / sqrt(dx * dx + dy * dy + dz * dz), total += weights[j] * dr and
// sum_array3d[i] += weights[j] * (dx, dy, dz); the output is
// (sum_array, sum_array3d), sum_array[i] the total of point i.
// dx * dx + dy * dy + dz * dz is a sum of squares, one reduction rescaled
// once (NPB_BASELINES.md section 5, rule 3: like the loops that accumulate
// tmp * tmp, here written out); its root is the exact floor square root and
// 1.0 / root a float division by a positive divisor (a zero distance has no
// proof: NumPy gives inf). total and each coordinate of sum_array3d[i]
// accumulate products into one variable from 0, one reduction each, rescaled
// once (rule 3).
// @outputs: float[NP]; float[NP * AXES]

use noir_hw::{check_inputs, divmod_pos, rescale, sqrt, SCALE};

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
                squares += d * d;
                sums[a] += weights[j] * d;
            }
            // 1.0 / root: a quotient floor(2^64 / root), root > 0 (divmod_pos).
            let (dr, _) = divmod_pos(SCALE * SCALE, sqrt(rescale(squares)));
            total += weights[j] * dr;
        }
        sum_array[i] = rescale(total);
        for a in 0..AXES {
            sum_array3d[i][a] = rescale(sums[a]);
        }
    }
    (sum_array, sum_array3d)
}
