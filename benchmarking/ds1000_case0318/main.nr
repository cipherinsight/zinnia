// DS-1000 case 0318 -- Noir baseline.
// c = inverse_permutation(permutation); a = a[:, c] on Float[8,6], perm[6].
// Binds the SAME contract as Zinnia: numpy index semantics accept
// permutation values in [-6, 6), so the scatter indicator is the dual
// equality (p == k) | (p == k - 6); and each gather index is enforced to
// hit exactly one column (one-hot membership assert) so an invalid index
// makes the proof fail instead of silently emitting a zero column.

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], permutation: [Field; 6]) -> pub [Field; 48] {
    // Inverse permutation: c[k] = j with perm[j] == k (mod COLS, numpy
    // negative-index semantics). Select-chain scatter form, mirroring
    // Zinnia's lowering.
    let mut c: [Field; 6] = [0; 6];
    for j in 0..COLS {
        let p = permutation[j];
        for k in 0..COLS {
            let hit = (p == k as Field) | (p + ((COLS - k) as Field) == 0);
            let is_eq: Field = if hit { 1 } else { 0 };
            c[k] = is_eq * (j as Field) + (1 - is_eq) * c[k];
        }
    }
    // Gather: out[i][j] = a[i][c[j]] via indicator vector, with a one-hot
    // membership assert per index.
    let mut out: [Field; 48] = [0; 48];
    for j in 0..COLS {
        let mut onehot: [Field; 6] = [0; 6];
        let mut total: Field = 0;
        for k in 0..COLS {
            let is_eq: Field = if c[j] == k as Field { 1 } else { 0 };
            onehot[k] = is_eq;
            total = total + is_eq;
        }
        assert(total == 1);
        for i in 0..ROWS {
            let mut v: Field = 0;
            for k in 0..COLS {
                v = v + a[i * COLS + k] * onehot[k];
            }
            out[i * COLS + j] = v;
        }
    }
    out
}
