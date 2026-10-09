// DS-1000 case 0434: literal translation of kernel.py (LITERAL_BASELINES.md).
// Normalize each NumPy index with wrap/bounds, then scan positions and select; complete row writes before column writes.
// @outputs: int[16]
fn main(a: [Field; 16], zero_rows: [Field; 2], zero_cols: [Field; 2]) -> pub [Field; 16] {
    noir_hw::check_inputs(a);
    let rows = zero_rows;
    let cols = zero_cols;
    noir_hw::check_inputs(rows);
    noir_hw::check_inputs(cols);
    let mut result = a;
    for i in 0..2 {
        let pos = noir_hw::index_pos::<8>(rows[i], 4);
        for row in 0..4 {
            let hit = pos == row;
            for col in 0..4 { result[row * 4 + col] = if hit { 0 } else { result[row * 4 + col] }; }
        }
    }
    for i in 0..2 {
        let pos = noir_hw::index_pos::<8>(cols[i], 4);
        for col in 0..4 {
            let hit = pos == col;
            for row in 0..4 { result[row * 4 + col] = if hit { 0 } else { result[row * 4 + col] }; }
        }
    }
    result
}
