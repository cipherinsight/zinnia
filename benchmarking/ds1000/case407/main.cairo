// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let a = array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32, 9_u32, 10_u32];

    // accmap = [0, 1, 0, 0, 0, 6, 8, 2, 2, 1]
    let accmap = array![0_u32, 1_u32, 0_u32, 0_u32, 0_u32, 6_u32, 8_u32, 2_u32, 2_u32, 1_u32];

    // result = [13, 12, 17]
    let result = array![13_u32, 12_u32, 17_u32];

    // Compute group sums for groups 0, 1, 2 only.
    let mut sum0: u32 = 0_u32;
    let mut sum1: u32 = 0_u32;
    let mut sum2: u32 = 0_u32;

    for i in 0_u32..10_u32 {
        let tag = *accmap.at(i);
        let val = *a.at(i);

        if tag == 0_u32 {
            sum0 = sum0 + val;
        }
        if tag == 1_u32 {
            sum1 = sum1 + val;
        }
        if tag == 2_u32 {
            sum2 = sum2 + val;
        }
    }

    let expected = array![sum0, sum1, sum2];

    // Assert result == expected
    for g in 0_u32..3_u32 {
        assert!(*result.at(g) == *expected.at(g));
    }
}
