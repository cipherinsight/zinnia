fn main(A: [[Field; 3]; 4], B: [[Field; 3]; 7], output: [[Field; 3]; 7]) {
    // --- Step 1: membership flags ---
    // inB[i] = true iff A[i] appears as a row in B
    let mut inB: [bool; 4] = [false, false, false, false];
    for i in 0..4 {
        let mut found: bool = false;
        for j in 0..7 {
            let m0 = A[i][0] == B[j][0];
            let m1 = A[i][1] == B[j][1];
            let m2 = A[i][2] == B[j][2];
            if m0 & m1 & m2 { found = true; }
        }
        inB[i] = found;
    }

    // inA[j] = true iff B[j] appears as a row in A
    let mut inA: [bool; 7] = [false, false, false, false, false, false, false];
    for j in 0..7 {
        let mut found: bool = false;
        for i in 0..4 {
            let m0 = B[j][0] == A[i][0];
            let m1 = B[j][1] == A[i][1];
            let m2 = B[j][2] == A[i][2];
            if m0 & m1 & m2 { found = true; }
        }
        inA[j] = found;
    }

    // --- Step 2: prefix counts for A-only and B-only ---
    // keep_A[i] = 1 if A[i] not in B else 0
    let mut keep_A: [Field; 4] = [0, 0, 0, 0];
    let mut prefA_before: [Field; 4] = [0, 0, 0, 0];
    let mut prefA: Field = 0;
    for i in 0..4 {
        prefA_before[i] = prefA;
        let not_inB: Field = if inB[i] { 0 } else { 1 };
        keep_A[i] = not_inB;
        prefA = prefA + not_inB;
    }
    // Exactly two rows A-only for this instance
    assert(prefA == 2, "Expected 2 rows from A not in B");

    // keep_B[j] = 1 if B[j] not in A else 0
    let mut keep_B: [Field; 7] = [0, 0, 0, 0, 0, 0, 0];
    let mut prefB_before: [Field; 7] = [0, 0, 0, 0, 0, 0, 0];
    let mut prefB: Field = 0;
    for j in 0..7 {
        prefB_before[j] = prefB;
        let not_inA: Field = if inA[j] { 0 } else { 1 };
        keep_B[j] = not_inA;
        prefB = prefB + not_inA;
    }
    // Exactly five rows B-only for this instance
    assert(prefB == 5, "Expected 5 rows from B not in A");

    // --- Step 3: construct expected symmetric difference ---
    // First two rows: A-only in A's order
    let mut exp: [[Field; 3]; 7] = [
        [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]
    ];
    for i in 0..4 {
        let is_keep = keep_A[i];
        let at_pos0: Field = if prefA_before[i] == Field::from(0) { 1 } else { 0 };
        let at_pos1: Field = if prefA_before[i] == Field::from(1) { 1 } else { 0 };
        let w0 = is_keep * at_pos0; // goes to exp[0]
        let w1 = is_keep * at_pos1; // goes to exp[1]
        for c in 0..3 {
            exp[0][c] = exp[0][c] + A[i][c] * w0;
            exp[1][c] = exp[1][c] + A[i][c] * w1;
        }
    }

    // Next five rows: B-only in B's order, placed at exp[2..7)
    for j in 0..7 {
        let is_keep = keep_B[j];
        // position r in {0..4} -> absolute row = 2 + r
        for r in 0..5 {
            let at_r: Field = if prefB_before[j] == Field::from(r as u128) { 1 } else { 0 };
            let w = is_keep * at_r;
            for c in 0..3 {
                exp[2 + r][c] = exp[2 + r][c] + B[j][c] * w;
            }
        }
    }

    // --- Step 4: compare ---
    for r in 0..7 {
        for c in 0..3 {
            assert(output[r][c] == exp[r][c], "Mismatch at output[{r}][{c}]");
        }
    }
}
