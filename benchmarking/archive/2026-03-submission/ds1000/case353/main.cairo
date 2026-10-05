// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are preserved as closely as Cairo allows.

#[executable]
pub fn main() {
    // A: 4x3
    let A = array![
        array![1_u32, 1_u32, 1_u32],
        array![1_u32, 1_u32, 2_u32],
        array![1_u32, 1_u32, 3_u32],
        array![1_u32, 1_u32, 4_u32],
    ];

    // B: 7x3
    let B = array![
        array![0_u32, 0_u32, 0_u32],
        array![1_u32, 0_u32, 2_u32],
        array![1_u32, 0_u32, 3_u32],
        array![1_u32, 0_u32, 4_u32],
        array![1_u32, 1_u32, 0_u32],
        array![1_u32, 1_u32, 1_u32],
        array![1_u32, 1_u32, 4_u32],
    ];

    // output: 2x3
    let output = array![
        array![1_u32, 1_u32, 2_u32],
        array![1_u32, 1_u32, 3_u32],
    ];

    // Step 1: For each row i of A, check membership in B (exact row match on 3 columns)
    let mut in_B = ArrayTrait::new(); // bool[4]
    for i in 0..4_u32 {
        let mut found: bool = false;
        for j in 0..7_u32 {
            let m0: bool = *A.at(i).at(0_u32) == *B.at(j).at(0_u32);
            let m1: bool = *A.at(i).at(1_u32) == *B.at(j).at(1_u32);
            let m2: bool = *A.at(i).at(2_u32) == *B.at(j).at(2_u32);
            let row_match: bool = (m0 && m1) && m2;
            if row_match {
                found = true;
            }
        }
        in_B.append(found);
    }

    // Step 2: prefix counts of rows NOT in B
    let mut pref: u32 = 0_u32;
    let mut pref_before = ArrayTrait::new(); // u32[4]
    let mut keep_flag   = ArrayTrait::new(); // u32[4], 1 if keep (not in B), else 0
    for i in 0..4_u32 {
        pref_before.append(pref);
        let mut not_in: u32 = 0_u32;
        if *in_B.at(i) {
            not_in = 0_u32;
        } else {
            not_in = 1_u32;
        }
        keep_flag.append(not_in);
        pref = pref + not_in;
    }
    // Exactly two rows should be kept for this instance
    assert!(pref == 2_u32);

    // Step 3: Build expected kept rows using indicators:
    // If keep_flag[i]==1 and pref_before[i]==0 -> goes to kept row 0
    // If keep_flag[i]==1 and pref_before[i]==1 -> goes to kept row 1
    // We accumulate the two kept rows' three columns in scalars.
    let mut exp0_c0: u32 = 0_u32;
    let mut exp0_c1: u32 = 0_u32;
    let mut exp0_c2: u32 = 0_u32;

    let mut exp1_c0: u32 = 0_u32;
    let mut exp1_c1: u32 = 0_u32;
    let mut exp1_c2: u32 = 0_u32;

    for i in 0..4_u32 {
        let is_keep: u32 = *keep_flag.at(i);

        let mut is_pos0: u32 = 0_u32;
        if *pref_before.at(i) == 0_u32 {
            is_pos0 = 1_u32;
        } else {
            is_pos0 = 0_u32;
        }

        let mut is_pos1: u32 = 0_u32;
        if *pref_before.at(i) == 1_u32 {
            is_pos1 = 1_u32;
        } else {
            is_pos1 = 0_u32;
        }

        let w0: u32 = is_keep * is_pos0;
        let w1: u32 = is_keep * is_pos1;

        // c = 0
        exp0_c0 = exp0_c0 + *A.at(i).at(0_u32) * w0;
        exp1_c0 = exp1_c0 + *A.at(i).at(0_u32) * w1;
        // c = 1
        exp0_c1 = exp0_c1 + *A.at(i).at(1_u32) * w0;
        exp1_c1 = exp1_c1 + *A.at(i).at(1_u32) * w1;
        // c = 2
        exp0_c2 = exp0_c2 + *A.at(i).at(2_u32) * w0;
        exp1_c2 = exp1_c2 + *A.at(i).at(2_u32) * w1;
    }

    // Step 4: Compare with provided output
    assert!(*output.at(0_u32).at(0_u32) == exp0_c0);
    assert!(*output.at(0_u32).at(1_u32) == exp0_c1);
    assert!(*output.at(0_u32).at(2_u32) == exp0_c2);

    assert!(*output.at(1_u32).at(0_u32) == exp1_c0);
    assert!(*output.at(1_u32).at(1_u32) == exp1_c1);
    assert!(*output.at(1_u32).at(2_u32) == exp1_c2);
}
