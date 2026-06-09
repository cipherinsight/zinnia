// DS-1000 case 441 — `number in a` on Integer[4].
// a = [3, 7, 11, 19], number = 11 → contained = 1.

#[executable]
pub fn main() {
    let a: Array<i128> = array![3, 7, 11, 19];
    let number: i128 = 11;
    let claimed: u128 = 1;

    let mut contained: u128 = 0;
    for k in 0_u32..4 {
        let eq: u128 = if *a.at(k) == number { 1 } else { 0 };
        contained = contained | eq;
    };

    assert!(contained == claimed);
}
