// pythran_periodic_bc__periodic_bc (Pythran periodic_bc), Noir baseline.
// Kernel: `f[:, 0, :] = f[:, -2, :]` on f: Float[N0, N1, N2], returning f.
// Data movement only: the plane f[:, 0, :] is overwritten before it is read,
// so it gets no domain check (unless it is the source plane itself); every
// other element is checked and output as given.
// @outputs: float[N0 * N1 * N2]

use noir_hw::check_input;

// Sizes (hw_check.sizes): from fixture.json.
global N0: u32 = 3; // = f_0
global N1: u32 = 3; // = f_1
global N2: u32 = 3; // = f_2
// The plane `-2` names (NumPy's wrap of a negative index).
global SOURCE: u32 = N1 - 2;

fn main(f: [[[Field; N2]; N1]; N0]) -> pub [[[Field; N2]; N1]; N0] {
    let mut result = f;
    for i in 0..N0 {
        for j in 0..N1 {
            // Input domain: every element the kernel reads is a Q32 code in
            // the window.
            if (j != 0) | (SOURCE == 0) {
                for k in 0..N2 {
                    check_input(f[i][j][k]);
                }
            }
        }
        for k in 0..N2 {
            result[i][0][k] = f[i][SOURCE][k];
        }
    }
    result
}
