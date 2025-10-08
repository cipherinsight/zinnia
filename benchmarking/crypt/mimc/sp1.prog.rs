#![no_main]
sp1_zkvm::entrypoint!(main);

use ethereum_types::U512;

// BN254 prime field modulus
const P_STR: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

fn add_mod(a: &U512, b: &U512, p: &U512) -> U512 {
    let mut res = (a + b) % *p;
    if res >= *p {
        res -= *p;
    }
    res
}

fn mul_mod(a: &U512, b: &U512, p: &U512) -> U512 {
    (a * b) % *p
}

fn pow3_mod(x: &U512, p: &U512) -> U512 {
    // (x^3) mod p
    let x2 = mul_mod(x, x, p);
    mul_mod(&x2, x, p)
}

// MiMC permutation with 8 rounds and fixed constants
fn mimc_permute(mut x: U512, p: &U512) -> U512 {
    let c = [
        U512::from(1u64),
        U512::from(2u64),
        U512::from(3u64),
        U512::from(4u64),
        U512::from(5u64),
        U512::from(6u64),
        U512::from(7u64),
        U512::from(8u64),
    ];

    for i in 0..8 {
        let tmp = add_mod(&x, &c[i], p);
        x = pow3_mod(&tmp, p);
    }
    x
}

// MiMC-3 hash of fixed 3-word message
fn mimc3_hash_3(msg: &[U512; 3], p: &U512) -> U512 {
    let mut state = U512::from(0u64);
    for i in 0..3 {
        let t = add_mod(&state, &msg[i], p);
        state = mimc_permute(t, p);
    }
    state
}

pub fn main() {
    // Read message words (3 field elements)
    let mut msg = [U512::from(0u64); 3];
    for i in 0..3 {
        msg[i] = sp1_zkvm::io::read::<U512>();
    }

    // Read expected hash value
    let expected = sp1_zkvm::io::read::<U512>();

    let p = U512::from_str_radix(P_STR, 10).unwrap();
    let h = mimc3_hash_3(&msg, &p);

    // Assert equality
    assert_eq!(h % p, expected % p);

    // Optionally commit result
    // sp1_zkvm::io::commit_slice(&h.to_le_bytes());
}
