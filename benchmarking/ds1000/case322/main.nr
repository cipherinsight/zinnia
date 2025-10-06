fn main(a: [[Field; 2]; 2], result: [[Field; 2]; 2]) {
    // Flatten in C order: p=0:(0,0), p=1:(0,1), p=2:(1,0), p=3:(1,1)
    let flat: [Field; 4] = [a[0][0], a[0][1], a[1][0], a[1][1]];

    // Compute min over the four entries
    let m01 = if flat[0].lt(flat[1]) { flat[0] } else { flat[1] };
    let m23 = if flat[2].lt(flat[3]) { flat[2] } else { flat[3] };
    let min_val = if m01.lt(m23) { m01 } else { m23 };

    // is_min[p] = [flat[p] == min_val]
    let is_min: [bool; 4] = [
        flat[0] == min_val,
        flat[1] == min_val,
        flat[2] == min_val,
        flat[3] == min_val,
    ];

    // prefix count of mins BEFORE position p (sum_prev[p])
    let mut pref: Field = 0;
    let mut sum_prev: [Field; 4] = [0, 0, 0, 0];
    for p in 0..4 {
        sum_prev[p] = pref;
        let ind: Field = if is_min[p] { 1 } else { 0 };
        pref = pref + ind;
    }

    // Coordinates for each flattened position p
    // p=0:(0,0)  p=1:(0,1)  p=2:(1,0)  p=3:(1,1)
    let prow: [Field; 4] = [0, 0, 1, 1];
    let pcol: [Field; 4] = [0, 1, 0, 1];

    // Build expected rows for rindex = 0,1 (first two occurrences of the min)
    // row_r = sum_p coords(p) * [is_min[p]] * [sum_prev[p] == r]
    let mut exp_r0_row: Field = 0;
    let mut exp_r0_col: Field = 0;
    let mut exp_r1_row: Field = 0;
    let mut exp_r1_col: Field = 0;

    for p in 0..4 {
        // Indicators for r=0 and r=1
        let eq0: bool = sum_prev[p] == Field::from(0);
        let eq1: bool = sum_prev[p] == Field::from(1);

        let ind_min: Field = if is_min[p] { 1 } else { 0 };
        let ind0: Field = if eq0 { 1 } else { 0 };
        let ind1: Field = if eq1 { 1 } else { 0 };

        // Contribute to row 0 (first min)
        let w0 = ind_min * ind0;
        exp_r0_row = exp_r0_row + prow[p] * w0;
        exp_r0_col = exp_r0_col + pcol[p] * w0;

        // Contribute to row 1 (second min)
        let w1 = ind_min * ind1;
        exp_r1_row = exp_r1_row + prow[p] * w1;
        exp_r1_col = exp_r1_col + pcol[p] * w1;
    }

    // Compare against provided result
    assert(result[0][0] == exp_r0_row, "Mismatch at (0,0)");
    assert(result[0][1] == exp_r0_col, "Mismatch at (0,1)");
    assert(result[1][0] == exp_r1_row, "Mismatch at (1,0)");
    assert(result[1][1] == exp_r1_col, "Mismatch at (1,1)");
}
