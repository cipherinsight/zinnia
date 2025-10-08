#![no_main]
use risc0_zkvm::guest::env;
use ethereum_types::U512;

// BN254 prime modulus
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
    let x2 = mul_mod(x, x, p);
    mul_mod(&x2, x, p)
}

// --- MiMC permutation ---
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

// --- MiMC-3 sponge hash ---
fn mimc3_hash_3(msg: &[U512; 3], p: &U512) -> U512 {
    let mut state = U512::from(0u64);
    for i in 0..3 {
        let t = add_mod(&state, &msg[i], p);
        state = mimc_permute(t, p);
    }
    state
}

risc0_zkvm::guest::entry!(main);
pub fn main() {
    // Read 3-word message
    let mut msg = [U512::from(0u64); 3];
    for i in 0..3 {
        msg[i] = env::read::<U512>();
    }

    // Read expected digest
    let expected = env::read::<U512>();

    // Compute MiMC-3 hash
    let p = U512::from_str_radix(P_STR, 10).unwrap();
    let h = mimc3_hash_3(&msg, &p);

    // Check equality
    assert_eq!(h % p, expected % p);

    // Optionally: env::commit(&h);
}
