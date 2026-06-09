// DS-1000 case 401 — `np.array([[], [], []])`. 3×0 empty array. Trivial.

#[executable]
pub fn main() {
    // Three empty rows; nothing to assert.
    let mut _rows: Array<Array<u128>> = ArrayTrait::new();
    _rows.append(ArrayTrait::new());
    _rows.append(ArrayTrait::new());
    _rows.append(ArrayTrait::new());
}
