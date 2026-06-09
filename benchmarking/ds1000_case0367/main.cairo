// DS-1000 case 367 — copy.deepcopy(array_of_arrays); pairs is unused.
// Mirrors target.rs: witness all 6 pair cells, then expose the 3 deepcopy outputs.

#[executable]
pub fn main() {
    let pairs: Array<i128> = array![1, 2, 3, 4, 5, 6];
    let array_of_arrays: Array<i128> = array![7, 8, 9];
    let expected: Array<i128> = array![7, 8, 9];

    // Witness `pairs` to mirror the IR (6 ReadInt) without using them.
    let mut i: u32 = 0;
    while i != 6 {
        let _ = *pairs.at(i);
        i = i + 1;
    };

    // Deep copy: independent array with the same values.
    let mut result: Array<i128> = ArrayTrait::new();
    let mut j: u32 = 0;
    while j != 3 {
        result.append(*array_of_arrays.at(j));
        j = j + 1;
    };

    let mut k: u32 = 0;
    while k != 3 {
        assert!(*result.at(k) == *expected.at(k));
        k = k + 1;
    };
}
