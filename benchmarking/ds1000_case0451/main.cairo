// DS-1000 case 451 — `np.zeros((20, 10, 10, 2))`.
// Zinnia represents the zero-constant output as a single public value
// (1× AllocateMemoryIR + 1× ConstantIntIR + 1× ExposePublic); we mirror that.

#[executable]
pub fn main() {
    let zero: u128 = 0;
    let claimed: u128 = 0;
    assert!(zero == claimed);
}
