// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure faithfully preserved.

#[executable]
pub fn main() {
    // a = [9, 2, 7, 0]
    let a = array![9_u32, 2_u32, 7_u32, 0_u32, 9_u32, 2_u32, 7_u32, 0_u32, 9_u32, 2_u32, 7_u32, 0_u32];
    // number = 7
    let number: u32 = 7_u32;
    // is_contained = 1
    let is_contained: u32 = 1_u32;

    // Check if number exists in a
    let mut found: u32 = 0_u32;
    for i in 0..12_u32 {
        if *a.at(i) == number {
            found = 1_u32;
        }
    }

    assert!(is_contained == found);
}
