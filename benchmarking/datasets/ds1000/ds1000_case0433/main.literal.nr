// DS-1000 case 0433: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each write tests both signed spellings of every axis position and enforces one hit; complete row writes before column writes.
// @outputs: int[16]
fn main(a: [Field; 16], zero_rows: Field, zero_cols: Field) -> pub [Field; 16] {
    noir_hw::check_inputs(a);
    let rows = [zero_rows];
    let cols = [zero_cols];
    noir_hw::check_inputs(rows);
    noir_hw::check_inputs(cols);
    let mut result = a;
    for i in 0..1 {
        let hits = write_hits(rows[i]);
        for row in 0..4 {
            let hit = hits[row];
            for col in 0..4 { result[row * 4 + col] = if hit { 0 } else { result[row * 4 + col] }; }
        }
    }
    for i in 0..1 {
        let hits = write_hits(cols[i]);
        for col in 0..4 {
            let hit = hits[col];
            for row in 0..4 { result[row * 4 + col] = if hit { 0 } else { result[row * 4 + col] }; }
        }
    }
    result
}

fn write_hits(idx: Field) -> [bool; 4] {
    let mut hits = [false; 4];
    let mut total: Field = 0;
    for k in 0..4 {
        hits[k] = (idx == k as Field) | (idx == (k as Field - 4));
        total += hits[k] as Field;
    }
    assert(total == 1);
    hits
}
