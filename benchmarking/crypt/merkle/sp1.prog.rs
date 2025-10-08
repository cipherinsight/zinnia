#![no_main]
sp1_zkvm::entrypoint!(main);

use ethereum_types::U512;

const P_STR: &str =
    "21888242871839275222246405745257275088548364400416034343698204186575808495617";

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

fn mimc3(x: &U512, k: &U512, p: &U512) -> U512 {
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
    let mut t = add_mod(x, k, p);
    for i in 0..8 {
        t = pow3_mod(&add_mod(&t, &c[i], p), p);
    }
    t
}

fn mimc_hash2(left: &U512, right: &U512, p: &U512) -> U512 {
    let s = add_mod(left, right, p);
    mimc3(&s, &U512::from(0u64), p)
}

fn merkle_root(leaves: &[U512; 8], p: &U512) -> U512 {
    let mut l1 = [U512::from(0u64); 4];
    for i in 0..4 {
        l1[i] = mimc_hash2(&leaves[2 * i], &leaves[2 * i + 1], p);
    }
    let mut l2 = [U512::from(0u64); 2];
    for i in 0..2 {
        l2[i] = mimc_hash2(&l1[2 * i], &l1[2 * i + 1], p);
    }
    mimc_hash2(&l2[0], &l2[1], p)
}

fn merkle_verify(leaf: &U512, path: &[U512; 3], bits: &[U512; 3], root: &U512, p: &U512) {
    let mut cur = leaf.clone();
    for d in 0..3 {
        if bits[d] == U512::from(0u64) {
            cur = mimc_hash2(&cur, &path[d], p);
        } else {
            cur = mimc_hash2(&path[d], &cur, p);
        }
    }
    assert_eq!(cur % p, *root % p);
}

pub fn main() {
    let mut leaves = [U512::from(0u64); 8];
    for i in 0..8 {
        leaves[i] = sp1_zkvm::io::read::<U512>();
    }
    let leaf_idx = sp1_zkvm::io::read::<U512>().as_usize();
    let mut path = [U512::from(0u64); 3];
    for i in 0..3 {
        path[i] = sp1_zkvm::io::read::<U512>();
    }
    let mut bits = [U512::from(0u64); 3];
    for i in 0..3 {
        bits[i] = sp1_zkvm::io::read::<U512>();
    }

    let p = U512::from_str_radix(P_STR, 10).unwrap();
    let root = merkle_root(&leaves, &p);
    let leaf = &leaves[leaf_idx];
    merkle_verify(leaf, &path, &bits, &root, &p);
}
