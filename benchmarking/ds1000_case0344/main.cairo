// DS-1000 case 344 — np.array(list(mystr), dtype=int) on Integer scalar.
// Zinnia lowers this to a passthrough of the scalar integer.

#[executable]
pub fn main() {
    let mystr: i128 = 12345;
    let claimed: i128 = 12345;

    let result: i128 = mystr;

    assert!(result == claimed);
}
