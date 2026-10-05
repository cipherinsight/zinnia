// DS-1000 case 0433, Noir baseline.
// Kernel: `a[zero_rows, :] = 0; a[:, zero_cols] = 0` on a: Integer[4, 4] with
// scalar indices.
// NumPy index rules: row i is hit when zero_rows is i or i - 4, tested once as
// (zero_rows - i)(zero_rows - i + 4) == 0, and exactly one row must be hit (an
// index outside [-4, 4) raises, so it has no proof); the same for columns.
// Each row and column test is made once.
// @outputs: int[16]

fn main(a: [Field; 16], zero_rows: Field, zero_cols: Field) -> pub [Field; 16] {
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

// Position p is hit when idx is p or p - 4; exactly one must be (NumPy's wrap
// and bounds for an index into an axis of length 4).
fn hits(idx: Field) -> [bool; 4] {
    let mut hit: [bool; 4] = [false; 4];
    let mut total: Field = 0;
    for p in 0..4 {
        let d = idx - p as Field;
        hit[p] = d * (d + 4) == 0;
        total += hit[p] as Field;
    }
    assert(total == 1, "index out of bounds");
    hit
}
