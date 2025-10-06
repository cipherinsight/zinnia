// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let a = array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32, 9_u32, 10_u32];

    // index = [0, 1, 0, 0, 0, 1, 1, 2, 2, 1]
    let index = array![0_u32, 1_u32, 0_u32, 0_u32, 0_u32, 1_u32, 1_u32, 2_u32, 2_u32, 1_u32];

    // result = [5, 10, 9]
    let result = array![5_u32, 10_u32, 9_u32];

    // Precompute expected max for each group statically
    let mut max0: u32 = 0_u32;
    let mut max1: u32 = 0_u32;
    let mut max2: u32 = 0_u32;

    for i in 0_u32..10_u32 {
        let idx = *index.at(i);
        let val = *a.at(i);

        if (idx == 0_u32) && (val > max0) {
            max0 = val;
        }
        if (idx == 1_u32) && (val > max1) {
            max1 = val;
        }
        if (idx == 2_u32) && (val > max2) {
            max2 = val;
        }
    }

    let expected = array![max0, max1, max2];

    // Assert result == expected
    for g in 0_u32..3_u32 {
        assert!(*result.at(g) == *expected.at(g));
    }
}
