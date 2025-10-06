fn main(data: [Field; 10], result: [Field; 3]) {
    // bin_size = 3
    // Trim to a multiple of 3: take first 9 elements, reshape to (3,3),
    // then take max along axis=1.

    // Row 0: data[0], data[1], data[2]
    let mut m0 = data[0];
    if m0.lt(data[1]) { m0 = data[1]; }
    if m0.lt(data[2]) { m0 = data[2]; }

    // Row 1: data[3], data[4], data[5]
    let mut m1 = data[3];
    if m1.lt(data[4]) { m1 = data[4]; }
    if m1.lt(data[5]) { m1 = data[5]; }

    // Row 2: data[6], data[7], data[8]
    let mut m2 = data[6];
    if m2.lt(data[7]) { m2 = data[7]; }
    if m2.lt(data[8]) { m2 = data[8]; }

    // Check expected = [m0, m1, m2]
    assert(result[0] == m0, "bin 0 max mismatch");
    assert(result[1] == m1, "bin 1 max mismatch");
    assert(result[2] == m2, "bin 2 max mismatch");
}
