// MiMC-3 over BN254 Field
// Round function: x <- (x + c[r])^3 for r = 0..7
// Sponge for 3-word input: state = 0; absorb each msg[i]; output final state

fn mimc_permute(mut x: Field) -> Field {
    let c = [1, 2, 3, 4, 5, 6, 7, 8];
    for r in 0..8 {
        let t = x + Field::from(c[r] as u128);
        x = t * t * t;
    }
    x
}

fn mimc3_hash_3(msg: [Field; 3]) -> Field {
    let mut state: Field = 0;
    state = mimc_permute(state + msg[0]);
    state = mimc_permute(state + msg[1]);
    state = mimc_permute(state + msg[2]);
    state
}

fn main(msg: [Field; 3], expected: Field) {
    let h = mimc3_hash_3(msg);
    assert(h == expected, "Hash mismatch");
}
