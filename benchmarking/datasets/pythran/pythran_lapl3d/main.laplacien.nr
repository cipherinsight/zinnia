// pythran_lapl3d__laplacien (Pythran lapl3d), Noir baseline.
// Kernel: on image: Float[H, W, C], the absolute 5-point Laplacian of the
// interior, out = |4 c - up - down - left - right| ((H - 2) x (W - 2) x C),
// divided by max(1.0, np.max(out)) + 1e-9.
// The stencil is exact; np.abs is `abs_int64`. The maximum is `max_of` over
// the magnitudes and the constant 1.0, which is max(1.0, np.max(out)) in one
// hint. Every magnitude is below the divisor, so each quotient lies in
// [0, 1): `div_fraction`, whose quotient check is 32 bits instead of the
// window (29,117 gates with `divmod_pos`). The four corner pixels reach no
// output, so they are not checked.
// @outputs: float[INTERIOR]

use noir_hw::{abs_int64, check_inputs, div_fraction, max_of, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global H: u32 = 16; // = image_0
global W: u32 = 16; // = image_1
global C: u32 = 3; // = image_2
global INTERIOR: u32 = (H - 2) * (W - 2) * C;

global EPSILON_Q: Field = 4; // round(1e-9 * 2^32)

fn main(image: [[[Field; C]; W]; H]) -> pub [Field; INTERIOR] {
    // Input domain: every pixel but the corners is a Q32 code in the window.
    for i in 0..H {
        for j in 0..W {
            if ((i != 0) & (i != H - 1)) | ((j != 0) & (j != W - 1)) {
                check_inputs(image[i][j]);
            }
        }
    }
    // The magnitudes, then 1.0 in the last slot.
    let mut candidates: [Field; INTERIOR + 1] = [0; INTERIOR + 1];
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let laplacian = 4 * image[i][j][k] - image[i - 1][j][k] - image[i + 1][j][k] - image[i][j - 1][k]
                    - image[i][j + 1][k];
                candidates[((i - 1) * (W - 2) + j - 1) * C + k] = abs_int64(laplacian);
            }
        }
    }
    candidates[INTERIOR] = SCALE;
    let divisor = max_of(candidates) + EPSILON_Q;
    let mut result: [Field; INTERIOR] = [0; INTERIOR];
    for t in 0..INTERIOR {
        result[t] = div_fraction(candidates[t], divisor);
    }
    result
}
