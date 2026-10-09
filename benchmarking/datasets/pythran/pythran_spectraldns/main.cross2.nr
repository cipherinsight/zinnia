// pythran_spectraldns__cross2 (Pythran spectraldns), Noir baseline.
// Kernel: cross1(c, a, b) writes the cross product of the real vector field
// a: Float[3, A1, A2, A3] and the complex one b: Complex[3, A1, A2, A3] into c,
// c[0] = a[0] * b[2] - a[2] * b[1], c[1] = a[2] * b[0] - a[0] * b[2],
// c[2] = a[0] * b[1] - a[1] * b[0]; then c *= 1j; returns c.
// The leading 3 is the cross product's, not a size. c is overwritten before it
// is read, so it is not a parameter here.
// A real times a complex value is two rounded products (the real and
// imaginary parts of the complex product with a zero imaginary part, each one
// rescale); NumPy materializes each product array before subtracting, so the
// products are rounded one by one, and a[0] * b[2], used twice, is computed
// once. * 1j maps x + yi to -y + xi exactly.
// @outputs: float[3 * 2 * POINTS]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.cross2.json.
global A1: u32 = 2; // = a_1
global A2: u32 = 5; // = a_2
global A3: u32 = 7; // = a_3
global POINTS: u32 = A1 * A2 * A3;

// The complex product of the real x and the complex y = [re, im].
fn real_times(x: Field, y: [Field; 2]) -> [Field; 2] {
    [mul(x, y[0]), mul(x, y[1])]
}

fn main(a: [[Field; POINTS]; 3], b: [[[Field; 2]; POINTS]; 3]) -> pub [[[Field; 2]; POINTS]; 3] {
    // Input domain: every input element is a Q32 code in the window.
    for k in 0..3 {
        check_inputs(a[k]);
        for p in 0..POINTS {
            check_inputs(b[k][p]);
        }
    }
    let mut c: [[[Field; 2]; POINTS]; 3] = [[[0; 2]; POINTS]; 3];
    for p in 0..POINTS {
        let a0b2 = real_times(a[0][p], b[2][p]);
        let a2b1 = real_times(a[2][p], b[1][p]);
        let a2b0 = real_times(a[2][p], b[0][p]);
        let a0b1 = real_times(a[0][p], b[1][p]);
        let a1b0 = real_times(a[1][p], b[0][p]);
        let terms = [(a0b2, a2b1), (a2b0, a0b2), (a0b1, a1b0)];
        for k in 0..3 {
            let (plus, minus) = terms[k];
            // (re + im i) * 1j = -im + re i
            c[k][p] = [minus[1] - plus[1], plus[0] - minus[0]];
        }
    }
    c
}
