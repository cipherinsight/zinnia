// DS-1000 case 0318 -- Noir baseline.
// c = inverse_permutation(permutation); a = a[:, c] on Float[8,6], perm[6].
// Approach: build c[k] = j such that perm[j]==k, then gather a[i][c[j]] per output cell.

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], permutation: [Field; 6]) -> pub [Field; 48] {
    // Inverse permutation: c[k] = j with perm[j] == k.
    let mut c: [Field; 6] = [0; 6];
    for k in 0..COLS {
        let mut acc: Field = 0;
        for j in 0..COLS {
            let is_eq: Field = if permutation[j] == k as Field { 1 } else { 0 };
            acc = acc + (j as Field) * is_eq;
        }
        c[k] = acc;
    }
    // Gather: out[i][j] = a[i][c[j]] via indicator vector.
    let mut out: [Field; 48] = [0; 48];
    for i in 0..ROWS {
        for j in 0..COLS {
            let mut v: Field = 0;
            for k in 0..COLS {
                let is_eq: Field = if c[j] == k as Field { 1 } else { 0 };
                v = v + a[i * COLS + k] * is_eq;
            }
            out[i * COLS + j] = v;
        }
    }
    out
}
