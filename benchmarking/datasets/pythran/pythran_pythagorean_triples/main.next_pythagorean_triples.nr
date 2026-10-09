// pythran_pythagorean_triples__next_pythagorean_triples (Pythran pythagorean_triples), Noir baseline.
// Kernel: the three children of each Pythagorean triple in
// `previous: Int[P, 3]` (Berggren's matrices, a constant 9 x 3 integer matrix
// of the kernel, stacked), reshaped to 3 P rows of 3.
// Output row 3 p + t is matrices 3 t .. 3 t + 2 applied to previous[p]: each
// value is a constant-coefficient integer combination of the triple, exact in
// the field.
// @outputs: int[P * 9]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global P: u32 = 1; // = previous_0
global COLS: u32 = 3; // = previous_1

// Berggren's matrices A, B, C, stacked (the kernel's constant).
global MATRICES: [[Field; 3]; 9] = [
    [-1, 2, 2], [-2, 1, 2], [-2, 2, 3],
    [1, 2, 2], [2, 1, 2], [2, 2, 3],
    [1, -2, 2], [2, -1, 2], [2, -2, 3],
];

fn main(previous: [[Field; COLS]; P]) -> pub [[Field; 9]; P] {
    // Input domain: every input element is an int64.
    for p in 0..P {
        check_inputs(previous[p]);
    }
    let mut result: [[Field; 9]; P] = [[0; 9]; P];
    for p in 0..P {
        for r in 0..9 {
            let mut acc: Field = 0;
            for c in 0..COLS {
                acc += MATRICES[r][c] * previous[p][c];
            }
            result[p][r] = acc;
        }
    }
    result
}
