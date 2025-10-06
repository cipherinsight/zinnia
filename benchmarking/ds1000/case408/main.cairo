// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure (two-pass seeded minima with existence checks) are faithfully preserved.

#[executable]
pub fn main() {
    // a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let a = array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32, 9_u32, 10_u32];

    // index = [0, 1, 0, 0, 0, 3, 4, 2, 2, 1]
    let index = array![0_u32, 1_u32, 0_u32, 0_u32, 0_u32, 3_u32, 4_u32, 2_u32, 2_u32, 1_u32];

    // result = [1, 2, 8]
    let result = array![1_u32, 2_u32, 8_u32];

    let n: u32 = 10_u32;

    // First pass: seed minima from the first occurrence of each group (0,1,2)
    let mut found0: bool = false;
    let mut found1: bool = false;
    let mut found2: bool = false;

    let mut min0: u32 = 0_u32;
    let mut min1: u32 = 0_u32;
    let mut min2: u32 = 0_u32;

    for i in 0_u32..n {
        let idx = *index.at(i);
        let val = *a.at(i);

        if (idx == 0_u32) && (!found0) {
            min0 = val;
            found0 = true;
        }
        if (idx == 1_u32) && (!found1) {
            min1 = val;
            found1 = true;
        }
        if (idx == 2_u32) && (!found2) {
            min2 = val;
            found2 = true;
        }
    }

    // Ensure each group exists
    assert!(found0 && found1 && found2);

    // Second pass: refine minima
    for i in 0_u32..n {
        let idx = *index.at(i);
        let val = *a.at(i);

        if (idx == 0_u32) && (val < min0) {
            min0 = val;
        }
        if (idx == 1_u32) && (val < min1) {
            min1 = val;
        }
        if (idx == 2_u32) && (val < min2) {
            min2 = val;
        }
    }

    // expected = [min0, min1, min2]
    let expected = array![min0, min1, min2];

    // Compare with provided result
    for g in 0_u32..3_u32 {
        assert!(*result.at(g) == *expected.at(g));
    }
}
