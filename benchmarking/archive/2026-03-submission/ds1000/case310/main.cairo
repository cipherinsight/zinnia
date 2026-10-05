// Cairo translation of the given Zinnia code.
// Inputs are hard-coded, logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a = [[10, 50, 30],
    //      [60, 20, 40]]
    let a = array![
        array![10_u32, 50_u32, 30_u32],
        array![60_u32, 20_u32, 40_u32],
    ];

    let result: u32 = 0_u32;

    // Flattened in C order: [10, 50, 30, 60, 20, 40]
    // Minimum value = 10 → raveled index = 0

    // Manual argmin computation
    let flat = array![10_u32, 50_u32, 30_u32, 60_u32, 20_u32, 40_u32];
    let mut min_val: u32 = *flat.at(0);
    let mut min_idx: u32 = 0_u32;

    for i in 1_u32..6_u32 {
        let val = *flat.at(i);
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }

    assert!(result == min_idx);
}
