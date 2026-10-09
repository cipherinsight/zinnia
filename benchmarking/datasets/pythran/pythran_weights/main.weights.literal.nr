// pythran_weights__weights: literal translation of kernel.py (LITERAL_BASELINES.md).
// Equality array and each reduction are separate; repeat threshold*length each iteration; float32 stays Q32.
// @outputs: float[S]

use noir_hw::{check_input, check_inputs, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global S: u32 = 5; // = input_data_0
global L: u32 = 2; // = input_data_1
global COUNT_BITS: u32 = 14; // = 14 * cdiv(bits(input_data_0), 14)

fn main(input_data: [[Field; L]; S], threshold: Field) -> pub [Field; S] {
    // Input domain: every input element is an int64 or a Q32 code in the window.
    for i in 0..S {
        check_inputs(input_data[i]);
    }
    check_input(threshold);
    let mut weights = [0; S];
    for i in 0..S {
        let vector = input_data[i];
        let mut matches = [[0; L]; S];
        for row in 0..S { for c in 0..L { matches[row][c] = (vector[c] == input_data[row][c]) as Field; } }
        let mut counts = [0; S];
        for row in 0..S { for c in 0..L { counts[row] += matches[row][c]; } }
        let bar = threshold * L as Field;
        let mut over = [0; S];
        for row in 0..S { over[row] = noir_hw::gt(counts[row]*SCALE,bar) as Field; }
        let mut total: Field = 0;
        for row in 0..S { total += over[row]; }
        weights[i] = noir_hw::div(SCALE,total*SCALE);
    }
    weights
}
