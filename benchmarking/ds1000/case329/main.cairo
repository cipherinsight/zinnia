// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

fn pow_u32(base: u32, power: u32) -> u32 {
    let mut acc: u32 = 1_u32;
    for _k in 0..power {
        acc = acc * base;
    }
    acc
}

#[executable]
pub fn main() {
    // a = [[0, 1],
    //      [2, 3]]
    let a = array![
        array![0_u32, 1_u32],
        array![2_u32, 3_u32],
    ];

    // power = 5
    let power: u32 = 5_u32;

    // result = [[0, 1],
    //           [32, 243]]
    let result = array![
        array![0_u32, 1_u32],
        array![32_u32, 243_u32],
    ];

    // computed = a ** power (element-wise)
    let mut computed = ArrayTrait::new();
    for i in 0..2_u32 {
        let mut row = ArrayTrait::new();
        for j in 0..2_u32 {
            let base: u32 = *a.at(i).at(j);
            let val: u32 = pow_u32(base, power);
            row.append(val);
        }
        computed.append(row);
    }

    // assert result == computed (element-wise)
    for i in 0..2_u32 {
        for j in 0..2_u32 {
            assert!(*result.at(i).at(j) == *computed.at(i).at(j));
        }
    }
}
