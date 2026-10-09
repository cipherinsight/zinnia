// pythran_lapl3d__laplacien: literal translation of kernel.py (LITERAL_BASELINES.md).
// Full array passes; integer 4 multiplication is exact; maximum scan then scalar maximum. Only unread corner inputs omitted.
// The max+epsilon divisor may cross signed64, as allowed by optimized division.
// @outputs: float[(H-2)*(W-2)*C]
global H: u32 = 16; // = image_0
global W: u32 = 16; // = image_1
global C: u32 = 3; // = image_2

fn main(image: [[[Field; C]; W]; H]) -> pub [Field; (H - 2) * (W - 2) * C] {
    for i in 0..H {
        for j in 0..W {
            if ((i != 0) & (i != H - 1)) | ((j != 0) & (j != W - 1)) {
                noir_hw::check_inputs(image[i][j]);
            }
        }
    }
    let mut values = [0; (H - 2) * (W - 2) * C];
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = 4 * image[i][j][k];
            }
        }
    }
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = values[at] - image[i - 1][j][k];
            }
        }
    }
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = values[at] - image[i + 1][j][k];
            }
        }
    }
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = values[at] - image[i][j - 1][k];
            }
        }
    }
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let at = ((i - 1) * (W - 2) + j - 1) * C + k;
                values[at] = values[at] - image[i][j + 1][k];
            }
        }
    }
    let mut out_image = [0; (H - 2) * (W - 2) * C];
    for t in 0..(H - 2) * (W - 2) * C {
        out_image[t] = noir_hw::abs(values[t]);
    }
    let mut valmax = out_image[0];
    for t in 1..(H - 2) * (W - 2) * C {
        valmax = noir_hw::max(valmax, out_image[t]);
    }
    valmax = noir_hw::max(noir_hw::SCALE, valmax) + 4;
    for t in 0..(H - 2) * (W - 2) * C {
        out_image[t] = noir_hw::canonical_div_nonzero(out_image[t], valmax);
    }
    out_image
}
