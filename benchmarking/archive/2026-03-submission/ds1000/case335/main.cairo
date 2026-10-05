// Cairo translation of the given Zinnia code.
// Inputs are hard-coded; logic and structure are faithfully preserved.

#[executable]
pub fn main() {
    // a = [10, 20, 30]
    // b = [30, 20, 20]
    // c = [50, 20, 40]
    let a = array![10_u32, 20_u32, 30_u32];
    let b = array![30_u32, 20_u32, 20_u32];
    let c = array![50_u32, 20_u32, 40_u32];

    // result = [50, 20, 40]
    let result = array![50_u32, 20_u32, 40_u32];

    // Reference formula: result = max([a, b, c], axis=0)
    let mut computed = ArrayTrait::new();
    for i in 0..3_u32 {
        let mut max_val: u32 = *a.at(i);
        let b_val: u32 = *b.at(i);
        let c_val: u32 = *c.at(i);

        if b_val > max_val {
            max_val = b_val;
        }
        if c_val > max_val {
            max_val = c_val;
        }

        computed.append(max_val);
    }

    // Verify result == computed
    for i in 0..3_u32 {
        assert!(*result.at(i) == *computed.at(i));
    }
}
