// ------------------------------------------------------------
// MiMC-3 over BN254 field (debug prints added)
// ------------------------------------------------------------
fn mimc3(mut x: Field, k: Field) -> Field {
    let c = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut t = x + k;
    for i in 0..8 {
        t = (t + Field::from(c[i] as u128));
        t = t * t * t;
    }
    t
}

fn mimc_hash2(left: Field, right: Field) -> Field {
    let res = mimc3(left + right, Field::from(0));
    res
}

// ------------------------------------------------------------
// Merkle root construction (depth = 3, with prints)
// ------------------------------------------------------------
fn merkle_root(leaves: [Field; 8]) -> Field {
    // Level 0 -> Level 1
    let mut L1: [Field; 4] = [0, 0, 0, 0];
    for i in 0..4 {
        L1[i] = mimc_hash2(leaves[2 * i], leaves[2 * i + 1]);
    }

    // Level 1 -> Level 2
    let mut L2: [Field; 2] = [0, 0];
    for i in 0..2 {
        L2[i] = mimc_hash2(L1[2 * i], L1[2 * i + 1]);
    }

    // Level 2 -> Root
    let root = mimc_hash2(L2[0], L2[1]);
    root
}

// ------------------------------------------------------------
// Merkle path verification (depth = 3, with prints)
// ------------------------------------------------------------
fn merkle_verify(leaf: Field, path: [Field; 3], index_bits: [Field; 3], root: Field) {
    let mut cur = leaf;
    for d in 0..3 {
        let bit = index_bits[d];
        let left  = mimc_hash2(cur, path[d]);
        let right = mimc_hash2(path[d], cur);
        cur = left * (1 - bit) + right * bit;
    }

    assert(cur == root, "Merkle verification failed");
}

// ------------------------------------------------------------
// Main circuit (with prints)
// ------------------------------------------------------------
fn main(
    leaves: [Field; 8],
    leaf_idx: Field,
    path: [Field; 3],
    bits: [Field; 3],
) {
    let root = merkle_root(leaves);

    // Select the leaf corresponding to leaf_idx
    let mut leaf: Field = 0;
    for i in 0..8 {
        let is_eq: Field = if leaf_idx == Field::from(i as u128) { 1 } else { 0 };
        leaf = leaf + leaves[i] * is_eq;
    }

    merkle_verify(leaf, path, bits, root);
}
