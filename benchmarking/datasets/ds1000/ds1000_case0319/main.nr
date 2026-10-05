// DS-1000 case 0319 -- Noir baseline.
// c = inverse_permutation(perm); result = a[c, :, :] on Float[8,6,8], perm[8].
// Mirrors Zinnia's lowering ALGORITHM: numpy index semantics accept
// permutation values in [-8, 8), so the scatter indicator is the dual
// equality (p == j) | (p == j - 8) applied as a select chain; each gather
// index is enforced to hit exactly one row (one-hot membership assert).

global D0: u32 = 8;
global D1: u32 = 6;
global D2: u32 = 8;

fn main(a: [Field; 384], permutation: [Field; 8]) -> pub [Field; 384] {
    // Inverse permutation as a select-chain scatter with dual-equality
    // indicators (numpy negative-index semantics).
    let mut c: [Field; 8] = [0; 8];
    for j in 0..D0 {
        let p = permutation[j];
        // Soundness: enforce numpy's valid-index window -D0 <= p < D0. The
        // membership assert below cannot catch an out-of-range p (the scattered
        // c[k] is always in [0, D0), so the one-hot always sums to 1 even when
        // an OOB p left a row at its default 0 -- a silent wrong row). Exact-
        // window bit-decomposition check (p + D0 in [0, 2*D0), 2*D0 = 16 <=
        // 2^4), matching Zinnia's discharge for sound-vs-sound parity.
        let sh = p + (D0 as Field);
        sh.assert_max_bit_size::<4>();
        ((2 * D0 - 1) as Field - sh).assert_max_bit_size::<4>();
        for k in 0..D0 {
            let hit = (p == k as Field) | (p + ((D0 - k) as Field) == 0);
            let is_eq: Field = if hit { 1 } else { 0 };
            c[k] = is_eq * (j as Field) + (1 - is_eq) * c[k];
        }
    }
    // Row gather: out[i] = a[c[i], :, :] via a one-hot indicator family
    // per index, with a membership assert.
    let mut out: [Field; 384] = [0; 384];
    for i in 0..D0 {
        let mut onehot: [Field; 8] = [0; 8];
        let mut total: Field = 0;
        for k in 0..D0 {
            let is_eq: Field = if c[i] == k as Field { 1 } else { 0 };
            onehot[k] = is_eq;
            total = total + is_eq;
        }
        assert(total == 1);
        for r in 0..D1 {
            for s in 0..D2 {
                let mut v: Field = 0;
                for k in 0..D0 {
                    v = v + a[k * (D1 * D2) + r * D2 + s] * onehot[k];
                }
                out[i * (D1 * D2) + r * D2 + s] = v;
            }
        }
    }
    out
}
