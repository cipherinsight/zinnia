// pythran_pythagorean_triples__next_pythagorean_triples: literal translation of kernel.py (LITERAL_BASELINES.md).
// Integer dot uses a running sum in contraction-index order; transposes and reshape move wires only.
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
    for p in 0..P { check_inputs(previous[p]); }
    let mut product = [[0; P]; 9];
    for r in 0..9 { for p in 0..P {
        let mut acc = 0;
        for c in 0..COLS { acc += MATRICES[r][c] * previous[p][c]; }
        product[r][p] = acc;
    } }
    let mut result = [[0; 9]; P];
    for p in 0..P { for r in 0..9 { result[p][r] = product[r][p]; } }
    result
}
