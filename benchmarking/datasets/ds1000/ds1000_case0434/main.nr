// DS-1000 case 0434, Noir baseline.
// Kernel: `a[zero_rows, :] = 0; a[:, zero_cols] = 0` on a: Integer[4, 4] with
// Integer[2] index arrays.
// NumPy index rules for each of the two indices: it hits row i when it is i or
// i - 4, tested once as (idx - i)(idx - i + 4) == 0, and exactly one row must
// be hit (an index outside [-4, 4) raises, so it has no proof); the same for
// columns. A row is zeroed when either index hits it; each row and column test
// is made once.
// @outputs: int[16]

fn main(a: [Field; 16], zero_rows: [Field; 2], zero_cols: [Field; 2]) -> pub [Field; 16] {
    let row_hit = hits(zero_rows);
    let col_hit = hits(zero_cols);
    let mut out: [Field; 16] = [0; 16];
    for i in 0..4 {
        for j in 0..4 {
            out[i * 4 + j] = if row_hit[i] | col_hit[j] { 0 } else { a[i * 4 + j] };
        }
    }
    out
}

// Position p is hit when some index is p or p - 4; each index must hit exactly
// one position (NumPy's wrap and bounds for an axis of length 4).
fn hits(idxs: [Field; 2]) -> [bool; 4] {
    let mut hit: [bool; 4] = [false; 4];
    for t in 0..2 {
        let mut total: Field = 0;
        for p in 0..4 {
            let d = idxs[t] - p as Field;
            let h = d * (d + 4) == 0;
            hit[p] = hit[p] | h;
            total += h as Field;
        }
        assert(total == 1, "index out of bounds");
    }
    hit
}
