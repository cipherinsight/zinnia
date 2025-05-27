use std::hash::poseidon;

fn poseidon10(inputs: [Field; 10]) -> Field {
    poseidon::bn254::hash_10(inputs)
}

fn main(
    a0: pub Field, a1: pub Field, a2: pub Field, a3: pub Field, a4: pub Field, a5: pub Field, a6: pub Field, a7: pub Field, a8: pub Field, a9: pub Field,
    expected: pub Field
) {
    let inputs: [Field; 10] = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9];

    let h = poseidon10(inputs);
    assert(h == expected, "Poseidon10 mismatch");
}
