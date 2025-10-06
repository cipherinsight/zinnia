// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure faithfully preserved.

#[executable]
pub fn main() {
    // a = [[10, 50, 30],
    //      [60, 20, 40]]
    let a = array![
        array![10_u32, 50_u32, 30_u32],
        array![60_u32, 20_u32, 40_u32],
    ];

    let result: u32 = 3_u32;

    // Flattened array (C order): [10, 50, 30, 60, 20, 40]
    // Maximum value = 60 → raveled index = 3
    let flat = array![10_u32, 50_u32, 30_u32, 60_u32, 20_u32, 40_u32];

    let mut max_val: u32 = *flat.at(0);
    let mut max_idx: u32 = 0_u32;

    for i in 1_u32..6_u32 {
        let val = *flat.at(i);
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    assert!(result == max_idx);
}
