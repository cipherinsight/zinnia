// pythran_spectraldns__add_pressure_diffusion (Pythran spectraldns), Noir baseline.
// Kernel: P_hat[:] = du0 * k_0 + du1 * k_1 + du2 * k_2 with du_k = dU[k]
// (dU: Complex[3, A1, A2, A3]) and k_k = K_over_K2[k] (Float[3, A1, A2, A3]),
// then a write of dU that reaches no output; returns P_hat
// (Complex[A1, A2, A3]).
// The leading 3 is the vector field's, not a size. Only dU and K_over_K2
// reach the output, so only they are parameters here (P_hat is overwritten
// first; U_hat, K2, K, nu feed the dead write).
// A complex times a real value is two rounded products (the real and
// imaginary parts of the complex product with a zero imaginary part, each one
// rescale); NumPy materializes the three product arrays before adding them, so
// each is rounded.
// @outputs: float[2 * POINTS]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.add_pressure_diffusion.json.
global A1: u32 = 2; // = dU_1
global A2: u32 = 5; // = dU_2
global A3: u32 = 7; // = dU_3
global POINTS: u32 = A1 * A2 * A3;

fn main(dU: [[[Field; 2]; POINTS]; 3], K_over_K2: [[Field; POINTS]; 3]) -> pub [[Field; 2]; POINTS] {
    // Input domain: every element of dU and K_over_K2 is a Q32 code in the window.
    for k in 0..3 {
        check_inputs(K_over_K2[k]);
        for p in 0..POINTS {
            check_inputs(dU[k][p]);
        }
    }
    let mut p_hat: [[Field; 2]; POINTS] = [[0; 2]; POINTS];
    for p in 0..POINTS {
        for part in 0..2 {
            let mut sum: Field = 0;
            for k in 0..3 {
                sum += mul(dU[k][p][part], K_over_K2[k][p]);
            }
            p_hat[p][part] = sum;
        }
    }
    p_hat
}
