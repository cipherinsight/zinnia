// pythran_periodic_bc__periodic_bc: literal translation of kernel.py (LITERAL_BASELINES.md).
// Slice copy; overwritten unread input plane gets no domain check, as required by statement plumbing.
// @outputs: float[N0 * N1 * N2]

use noir_hw::check_input;

// Sizes (hw_check.sizes): from fixture.json.
global N0: u32 = 3; // = f_0
global N1: u32 = 3; // = f_1
global N2: u32 = 3; // = f_2
// The plane `-2` names (NumPy's wrap of a negative index).
global SOURCE: u32 = N1 - 2;

fn main(f: [[[Field; N2]; N1]; N0]) -> pub [[[Field; N2]; N1]; N0] {
    for i in 0..N0 {
        for j in 0..N1 {
            if (j != 0) | (SOURCE == 0) {
                for k in 0..N2 {
                    check_input(f[i][j][k]);
                }
            }
        }
    }
    let mut source = [[0; N2]; N0];
    for i in 0..N0 {
        for k in 0..N2 {
            source[i][k] = f[i][SOURCE][k];
        }
    }
    let mut result = f;
    for i in 0..N0 {
        for k in 0..N2 {
            result[i][0][k] = source[i][k];
        }
    }
    result
}
