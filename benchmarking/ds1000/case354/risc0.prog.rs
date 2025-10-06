// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read A (4x3)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..4 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read::<i32>());
        }
        a.push(row);
    }

    // read B (7x3)
    let mut b: Vec<Vec<i32>> = Vec::new();
    for _ in 0..7 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read::<i32>());
        }
        b.push(row);
    }

    // read output (7x3)
    let mut output: Vec<Vec<i32>> = Vec::new();
    for _ in 0..7 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..3 {
            row.push(env::read::<i32>());
        }
        output.push(row);
    }

    // --- Step 1: membership flags ---
    // inB[i] = true iff A[i] appears as a row in B
    let mut in_b: Vec<bool> = vec![false; 4];
    for i in 0..4 {
        let mut found = false;
        for j in 0..7 {
            let m0 = a[i as usize][0] == b[j as usize][0];
            let m1 = a[i as usize][1] == b[j as usize][1];
            let m2 = a[i as usize][2] == b[j as usize][2];
            if m0 && m1 && m2 {
                found = true;
            }
        }
        in_b[i as usize] = found;
    }

    // inA[j] = true iff B[j] appears as a row in A
    let mut in_a: Vec<bool> = vec![false; 7];
    for j in 0..7 {
        let mut found = false;
        for i in 0..4 {
            let m0 = b[j as usize][0] == a[i as usize][0];
            let m1 = b[j as usize][1] == a[i as usize][1];
            let m2 = b[j as usize][2] == a[i as usize][2];
            if m0 && m1 && m2 {
                found = true;
            }
        }
        in_a[j as usize] = found;
    }

    // --- Step 2: prefix counts for A-only and B-only ---
    // keep_A[i] = 1 if A[i] not in B else 0
    let mut keep_a: [i32; 4] = [0, 0, 0, 0];
    let mut pref_a_before: [i32; 4] = [0, 0, 0, 0];
    let mut pref_a: i32 = 0;
    for i in 0..4 {
        pref_a_before[i as usize] = pref_a;
        let not_in_b: i32 = if in_b[i as usize] { 0 } else { 1 };
        keep_a[i as usize] = not_in_b;
        pref_a = pref_a + not_in_b;
    }
    assert!(pref_a == 2);

    // keep_B[j] = 1 if B[j] not in A else 0
    let mut keep_b: [i32; 7] = [0; 7];
    let mut pref_b_before: [i32; 7] = [0; 7];
    let mut pref_b: i32 = 0;
    for j in 0..7 {
        pref_b_before[j as usize] = pref_b;
        let not_in_a: i32 = if in_a[j as usize] { 0 } else { 1 };
        keep_b[j as usize] = not_in_a;
        pref_b = pref_b + not_in_a;
    }
    assert!(pref_b == 5);

    // --- Step 3: construct expected symmetric difference ---
    let mut exp: Vec<Vec<i32>> = vec![vec![0; 3]; 7];

    // First two rows: A-only in A's order
    for i in 0..4 {
        let is_keep = keep_a[i as usize];
        let at_pos0: i32 = if pref_a_before[i as usize] == 0 { 1 } else { 0 };
        let at_pos1: i32 = if pref_a_before[i as usize] == 1 { 1 } else { 0 };
        let w0 = is_keep * at_pos0;
        let w1 = is_keep * at_pos1;
        for c in 0..3 {
            exp[0][c as usize] = exp[0][c as usize] + a[i as usize][c as usize] * w0;
            exp[1][c as usize] = exp[1][c as usize] + a[i as usize][c as usize] * w1;
        }
    }

    // Next five rows: B-only in B's order, placed at exp[2..7)
    for j in 0..7 {
        let is_keep = keep_b[j as usize];
        for r in 0..5 {
            let at_r: i32 = if pref_b_before[j as usize] == r as i32 { 1 } else { 0 };
            let w = is_keep * at_r;
            for c in 0..3 {
                exp[(2 + r) as usize][c as usize] =
                    exp[(2 + r) as usize][c as usize] + b[j as usize][c as usize] * w;
            }
        }
    }

    // --- Step 4: compare ---
    for r in 0..7 {
        for c in 0..3 {
            assert_eq!(output[r as usize][c as usize], exp[r as usize][c as usize]);
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
