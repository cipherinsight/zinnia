// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read A (4x3)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..4 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read());
        }
        a.push(row);
    }

    // Read B (7x3)
    let mut b: Vec<Vec<i32>> = Vec::new();
    for _ in 0..7 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read());
        }
        b.push(row);
    }

    // Read output (2x3)
    let mut output: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read());
        }
        output.push(row);
    }

    // Step 1: membership of each row of A in B (exact match on 3 columns)
    let mut in_b: Vec<bool> = vec![false; 4];
    for i in 0..4 {
        let mut found: bool = false;
        for j in 0..7 {
            let m0 = a[i as usize][0] == b[j as usize][0];
            let m1 = a[i as usize][1] == b[j as usize][1];
            let m2 = a[i as usize][2] == b[j as usize][2];
            let row_match = m0 && m1 && m2;
            if row_match {
                found = true; // no early break (preserve original logic)
            }
        }
        in_b[i as usize] = found;
    }

    // Step 2: prefix counts of rows NOT in B
    let mut pref: i32 = 0;
    let mut pref_before: Vec<i32> = vec![0; 4];
    let mut keep_flag: Vec<i32> = vec![0; 4];

    for i in 0..4 {
        pref_before[i as usize] = pref;
        let not_in: i32 = if in_b[i as usize] { 0 } else { 1 };
        keep_flag[i as usize] = not_in;
        pref = pref + not_in;
    }

    // Exactly two rows should be kept
    assert_eq!(pref, 2);

    // Step 3: build expected kept rows using indicators
    let mut exp: Vec<Vec<i32>> = vec![vec![0; 3], vec![0; 3]];
    for i in 0..4 {
        let is_keep: i32 = keep_flag[i as usize];

        let is_pos0: i32 = if pref_before[i as usize] == 0 { 1 } else { 0 };
        let is_pos1: i32 = if pref_before[i as usize] == 1 { 1 } else { 0 };

        let w0 = is_keep * is_pos0;
        let w1 = is_keep * is_pos1;

        for c in 0..3 {
            exp[0][c as usize] = exp[0][c as usize] + a[i as usize][c as usize] * w0;
            exp[1][c as usize] = exp[1][c as usize] + a[i as usize][c as usize] * w1;
        }
    }

    // Step 4: compare with provided output
    for r in 0..2 {
        for c in 0..3 {
            assert_eq!(output[r as usize][c as usize], exp[r as usize][c as usize]);
        }
    }

    // env::commit(&output);
}
