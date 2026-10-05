fn main(A: [[Field; 3]; 4], B: [[Field; 3]; 7], output: [[Field; 3]; 2]) {
    // Step 1: For each row i of A, check membership in B (exact row match on 3 columns)
    let mut in_B: [bool; 4] = [false, false, false, false];

    for i in 0..4 {
        let mut found: bool = false;
        for j in 0..7 {
            let m0 = A[i][0] == B[j][0];
            let m1 = A[i][1] == B[j][1];
            let m2 = A[i][2] == B[j][2];
            let row_match = m0 & m1 & m2;
            // OR-accumulate
            if row_match {
                found = true;
            }
        }
        in_B[i] = found;
    }

    // Step 2: prefix counts of rows NOT in B, to determine positions in the kept list
    let mut pref: Field = 0;                  // number of kept rows seen so far
    let mut pref_before: [Field; 4] = [0, 0, 0, 0];
    let mut keep_flag: [Field; 4] = [0, 0, 0, 0];  // 1 if A[i] not in B else 0

    for i in 0..4 {
        pref_before[i] = pref;

        let not_in: Field = if in_B[i] { 0 } else { 1 };
        keep_flag[i] = not_in;
        pref = pref + not_in;
    }

    // Exactly two rows should be kept for this instance
    assert(pref == 2, "Expected exactly two rows not in B");

    // Step 3: Build expected kept rows using indicators:
    // If keep_flag[i]==1 and pref_before[i]==0 -> goes to kept row 0
    // If keep_flag[i]==1 and pref_before[i]==1 -> goes to kept row 1
    let mut exp: [[Field; 3]; 2] = [[0, 0, 0], [0, 0, 0]];
    for i in 0..4 {
        let is_keep = keep_flag[i];

        let is_pos0: Field = {
            let eq0: bool = pref_before[i] == Field::from(0);
            if eq0 { 1 } else { 0 }
        };
        let is_pos1: Field = {
            let eq1: bool = pref_before[i] == Field::from(1);
            if eq1 { 1 } else { 0 }
        };

        let w0 = is_keep * is_pos0;
        let w1 = is_keep * is_pos1;

        for c in 0..3 {
            exp[0][c] = exp[0][c] + A[i][c] * w0;
            exp[1][c] = exp[1][c] + A[i][c] * w1;
        }
    }

    // Step 4: Compare with provided output
    for r in 0..2 {
        for c in 0..3 {
            assert(output[r][c] == exp[r][c], "Mismatch at output[{r}][{c}]");
        }
    }
}
