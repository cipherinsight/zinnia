// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a =
    // [[1, 0],
    //  [0, 2]]
    let a = array![
        array![1_u32, 0_u32],
        array![0_u32, 2_u32],
    ];

    // Expected result =
    // [[0, 1],
    //  [1, 0]]
    let result = array![
        array![0_u32, 1_u32],
        array![1_u32, 0_u32],
    ];

    // Step 1: Compute the minimum value
    let mut min_val: u32 = *a.at(0_u32).at(0_u32);
    for i in 0_u32..2_u32 {
        for j in 0_u32..2_u32 {
            let v: u32 = *a.at(i).at(j);
            if v < min_val {
                min_val = v;
            }
        }
    }

    // Step 2: Collect indices where a[i, j] == min_val
    // Since argwhere is unavailable, we manually build result
    let mut expected = ArrayTrait::new();
    let mut idx: u32 = 0_u32;
    for i in 0_u32..2_u32 {
        for j in 0_u32..2_u32 {
            if *a.at(i).at(j) == min_val {
                let mut row = ArrayTrait::new();
                row.append(i);
                row.append(j);
                expected.append(row);
                idx += 1_u32;
            }
        }
    }

    // Step 3: Verify the result
    // We know there are exactly 2 minima for this input.
    assert!(idx == 2_u32);
    for r in 0_u32..2_u32 {
        for c in 0_u32..2_u32 {
            assert!(*result.at(r).at(c) == *expected.at(r).at(c));
        }
    }
}
