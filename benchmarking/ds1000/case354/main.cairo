// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

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

    // output: 7x3
    let output = array![
        array![1_u32, 1_u32, 2_u32],
        array![1_u32, 1_u32, 3_u32],
        array![0_u32, 0_u32, 0_u32],
        array![1_u32, 0_u32, 2_u32],
        array![1_u32, 0_u32, 3_u32],
        array![1_u32, 0_u32, 4_u32],
        array![1_u32, 1_u32, 0_u32],
    ];

    // --- Step 1: membership flags ---
    // inB[i] = true iff A[i] appears as a row in B
    let mut inB = ArrayTrait::new(); // bool[4]
    for i in 0..4_u32 {
        let mut found: bool = false;
        for j in 0..7_u32 {
            let m0 = *A.at(i).at(0_u32) == *B.at(j).at(0_u32);
            let m1 = *A.at(i).at(1_u32) == *B.at(j).at(1_u32);
            let m2 = *A.at(i).at(2_u32) == *B.at(j).at(2_u32);
            if (m0 && m1) && m2 {
                found = true;
            }
        }
        inB.append(found);
    }

    // inA[j] = true iff B[j] appears as a row in A
    let mut inA = ArrayTrait::new(); // bool[7]
    for j in 0..7_u32 {
        let mut found: bool = false;
        for i in 0..4_u32 {
            let m0 = *B.at(j).at(0_u32) == *A.at(i).at(0_u32);
            let m1 = *B.at(j).at(1_u32) == *A.at(i).at(1_u32);
            let m2 = *B.at(j).at(2_u32) == *A.at(i).at(2_u32);
            if (m0 && m1) && m2 {
                found = true;
            }
        }
        inA.append(found);
    }

    // --- Step 2: prefix counts for A-only and B-only ---
    // keep_A[i] = 1 if A[i] not in B else 0
    let mut keep_A = ArrayTrait::new();       // u32[4]
    let mut prefA_before = ArrayTrait::new(); // u32[4]
    let mut prefA: u32 = 0_u32;
    for i in 0..4_u32 {
        prefA_before.append(prefA);
        let mut not_inB: u32 = 0_u32;
        if *inB.at(i) {
            not_inB = 0_u32;
        } else {
            not_inB = 1_u32;
        }
        keep_A.append(not_inB);
        prefA = prefA + not_inB;
    }
    // Exactly two rows A-only
    assert!(prefA == 2_u32);

    // keep_B[j] = 1 if B[j] not in A else 0
    let mut keep_B = ArrayTrait::new();       // u32[7]
    let mut prefB_before = ArrayTrait::new(); // u32[7]
    let mut prefB: u32 = 0_u32;
    for j in 0..7_u32 {
        prefB_before.append(prefB);
        let mut not_inA: u32 = 0_u32;
        if *inA.at(j) {
            not_inA = 0_u32;
        } else {
            not_inA = 1_u32;
        }
        keep_B.append(not_inA);
        prefB = prefB + not_inA;
    }
    // Exactly five rows B-only
    assert!(prefB == 5_u32);

    // --- Step 3: construct expected symmetric difference ---
    // We'll accumulate into scalars for exp[0..6][0..2].
    let mut exp0_c0: u32 = 0_u32; let mut exp0_c1: u32 = 0_u32; let mut exp0_c2: u32 = 0_u32;
    let mut exp1_c0: u32 = 0_u32; let mut exp1_c1: u32 = 0_u32; let mut exp1_c2: u32 = 0_u32;
    let mut exp2_c0: u32 = 0_u32; let mut exp2_c1: u32 = 0_u32; let mut exp2_c2: u32 = 0_u32;
    let mut exp3_c0: u32 = 0_u32; let mut exp3_c1: u32 = 0_u32; let mut exp3_c2: u32 = 0_u32;
    let mut exp4_c0: u32 = 0_u32; let mut exp4_c1: u32 = 0_u32; let mut exp4_c2: u32 = 0_u32;
    let mut exp5_c0: u32 = 0_u32; let mut exp5_c1: u32 = 0_u32; let mut exp5_c2: u32 = 0_u32;
    let mut exp6_c0: u32 = 0_u32; let mut exp6_c1: u32 = 0_u32; let mut exp6_c2: u32 = 0_u32;

    // First two rows: A-only, in A's order
    for i in 0..4_u32 {
        let is_keep: u32 = *keep_A.at(i);

        let mut at_pos0: u32 = 0_u32;
        if *prefA_before.at(i) == 0_u32 { at_pos0 = 1_u32; }

        let mut at_pos1: u32 = 0_u32;
        if *prefA_before.at(i) == 1_u32 { at_pos1 = 1_u32; }

        let w0 = is_keep * at_pos0; // goes to exp[0]
        let w1 = is_keep * at_pos1; // goes to exp[1]

        exp0_c0 = exp0_c0 + *A.at(i).at(0_u32) * w0;
        exp0_c1 = exp0_c1 + *A.at(i).at(1_u32) * w0;
        exp0_c2 = exp0_c2 + *A.at(i).at(2_u32) * w0;

        exp1_c0 = exp1_c0 + *A.at(i).at(0_u32) * w1;
        exp1_c1 = exp1_c1 + *A.at(i).at(1_u32) * w1;
        exp1_c2 = exp1_c2 + *A.at(i).at(2_u32) * w1;
    }

    // Next five rows: B-only, in B's order, placed at exp[2..6]
    for j in 0..7_u32 {
        let is_keep: u32 = *keep_B.at(j);
        // position r in {0..4} -> absolute row = 2 + r
        for r in 0..5_u32 {
            let mut at_r: u32 = 0_u32;
            if *prefB_before.at(j) == r { at_r = 1_u32; }
            let w = is_keep * at_r;

            let bj0 = *B.at(j).at(0_u32);
            let bj1 = *B.at(j).at(1_u32);
            let bj2 = *B.at(j).at(2_u32);

            if r == 0_u32 {
                exp2_c0 = exp2_c0 + bj0 * w;
                exp2_c1 = exp2_c1 + bj1 * w;
                exp2_c2 = exp2_c2 + bj2 * w;
            } else if r == 1_u32 {
                exp3_c0 = exp3_c0 + bj0 * w;
                exp3_c1 = exp3_c1 + bj1 * w;
                exp3_c2 = exp3_c2 + bj2 * w;
            } else if r == 2_u32 {
                exp4_c0 = exp4_c0 + bj0 * w;
                exp4_c1 = exp4_c1 + bj1 * w;
                exp4_c2 = exp4_c2 + bj2 * w;
            } else if r == 3_u32 {
                exp5_c0 = exp5_c0 + bj0 * w;
                exp5_c1 = exp5_c1 + bj1 * w;
                exp5_c2 = exp5_c2 + bj2 * w;
            } else {
                // r == 4
                exp6_c0 = exp6_c0 + bj0 * w;
                exp6_c1 = exp6_c1 + bj1 * w;
                exp6_c2 = exp6_c2 + bj2 * w;
            }
        }
    }

    // --- Step 4: compare ---
    // Row 0
    assert!(*output.at(0_u32).at(0_u32) == exp0_c0);
    assert!(*output.at(0_u32).at(1_u32) == exp0_c1);
    assert!(*output.at(0_u32).at(2_u32) == exp0_c2);
    // Row 1
    assert!(*output.at(1_u32).at(0_u32) == exp1_c0);
    assert!(*output.at(1_u32).at(1_u32) == exp1_c1);
    assert!(*output.at(1_u32).at(2_u32) == exp1_c2);
    // Row 2
    assert!(*output.at(2_u32).at(0_u32) == exp2_c0);
    assert!(*output.at(2_u32).at(1_u32) == exp2_c1);
    assert!(*output.at(2_u32).at(2_u32) == exp2_c2);
    // Row 3
    assert!(*output.at(3_u32).at(0_u32) == exp3_c0);
    assert!(*output.at(3_u32).at(1_u32) == exp3_c1);
    assert!(*output.at(3_u32).at(2_u32) == exp3_c2);
    // Row 4
    assert!(*output.at(4_u32).at(0_u32) == exp4_c0);
    assert!(*output.at(4_u32).at(1_u32) == exp4_c1);
    assert!(*output.at(4_u32).at(2_u32) == exp4_c2);
    // Row 5
    assert!(*output.at(5_u32).at(0_u32) == exp5_c0);
    assert!(*output.at(5_u32).at(1_u32) == exp5_c1);
    assert!(*output.at(5_u32).at(2_u32) == exp5_c2);
    // Row 6
    assert!(*output.at(6_u32).at(0_u32) == exp6_c0);
    assert!(*output.at(6_u32).at(1_u32) == exp6_c1);
    assert!(*output.at(6_u32).at(2_u32) == exp6_c2);
}
