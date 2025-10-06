// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure faithfully preserved.

#[executable]
pub fn main() {
    // x = [0, 1, 1, 1, 3, 1, 5, 5, 5]
    let x = array![0_u32, 1_u32, 1_u32, 1_u32, 3_u32, 1_u32, 5_u32, 5_u32, 5_u32];
    // y = [0, 2, 3, 4, 2, 4, 3, 4, 5]
    let y = array![0_u32, 2_u32, 3_u32, 4_u32, 2_u32, 4_u32, 3_u32, 4_u32, 5_u32];
    // (a, b) = (1, 4)
    let a: u32 = 1_u32;
    let b: u32 = 4_u32;
    // result = 3
    let result: u32 = 3_u32;

    // We search for the first index i such that x[i] == a and y[i] == b
    // Expected result: 3
    let n: u32 = 9_u32;
    let mut found_index: i32 = -1_i32;

    for i in 0..n {
        let xi = *x.at(i);
        let yi = *y.at(i);
        let cond1: bool = xi == a;
        let cond2: bool = yi == b;
        let cond3: bool = found_index == -1_i32;

        if cond1 && cond2 && cond3 {
            found_index = i.try_into().unwrap();
        }
    }

    let expected: i32 = found_index;
    assert!(result == expected.try_into().unwrap());
}
