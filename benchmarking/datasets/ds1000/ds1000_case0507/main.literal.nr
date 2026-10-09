// DS-1000 case 0507: literal translation of kernel.py (LITERAL_BASELINES.md).
// Coordinate arrays use prefix-count/equality compaction, then independent min/max scans.
// Dynamic loops run to the input shape with active flags; empty branches use safe dummy reads.
// @outputs: dynamic2 int[5x6]
global R: u32 = 5;
global C: u32 = 6;
fn main(im: [Field; R*C]) -> pub [Field; 2+R*C] {
    noir_hw::check_inputs(im);
    let mut mask = [false; R*C];
    for i in 0..R*C { mask[i] = im[i] == 0; }
    let mut inverted = [false; R*C];
    for i in 0..R*C { inverted[i] = !mask[i]; }
    let mut sums = [0; 5];
    for i in 0..5 { for j in 0..6 { sums[i] += inverted[i*6+j] as Field; } }
    let mut keep = [false; 5]; let mut payload = [[0; 1]; 5];
    for i in 0..5 { keep[i] = sums[i] != 0; payload[i] = [i as Field]; }
    let (rows_count, coords) = nonzero(keep, payload);
    let mut rows = [0; 5];
    for i in 0..5 { rows[i] = coords[i][0]; }
    let mut inverted = [false; R*C];
    for i in 0..R*C { inverted[i] = !mask[i]; }
    let mut sums = [0; 6];
    for i in 0..6 { for j in 0..5 { sums[i] += inverted[j*6+i] as Field; } }
    let mut keep = [false; 6]; let mut payload = [[0; 1]; 6];
    for i in 0..6 { keep[i] = sums[i] != 0; payload[i] = [i as Field]; }
    let (cols_count, coords) = nonzero(keep, payload);
    let mut cols = [0; 6];
    for i in 0..6 { cols[i] = coords[i][0]; }
    let present = rows_count != 0;
    let top = reduce(rows, rows_count, false);
    let bottom = reduce(rows, rows_count, true)+1;
    let left = reduce(cols, cols_count, false);
    let right = reduce(cols, cols_count, true)+1;
    crop(im, top, bottom, left, right, present)
}

// L5 compaction of coordinate tuples: one equality test per item and slot.
fn nonzero<let N: u32, let U: u32>(keep: [bool; N], payload: [[Field; U]; N]) -> (Field, [[Field; U]; N]) {
    let mut before = [0; N]; let mut count: Field = 0;
    for i in 0..N { before[i] = count; count += keep[i] as Field; }
    let mut result = [[0; U]; N];
    for j in 0..N {
        for i in 0..N {
            let hit = (keep[i] & (before[i] == j as Field)) as Field;
            for u in 0..U { result[j][u] += hit * payload[i][u]; }
        }
    }
    (count, result)
}
fn reduce<let N: u32>(xs: [Field; N], count: Field, maximum: bool) -> Field {
    let mut acc = xs[0];
    for i in 1..N {
        let active = noir_hw::lt(i as Field, count);
        let candidate = if maximum { noir_hw::max(acc, xs[i]) } else { noir_hw::min(acc, xs[i]) };
        acc = if active { candidate } else { acc };
    }
    acc
}
fn crop(image: [Field; R*C], top: Field, bottom: Field, left: Field, right: Field, present: bool) -> [Field; 2+R*C] {
    let height = if present { bottom-top } else { 0 };
    let width = if present { right-left } else { 0 };
    let size = height*width;
    let mut result = [0; 2+R*C]; result[0] = height; result[1] = width;
    for s in 0..R*C {
        let active = noir_hw::lt(s as Field, size);
        let safe_width = if present { width } else { 1 };
        let (dy, dx) = noir_hw::divmod_pos(s as Field, safe_width);
        let index = (top+dy)*(C as Field)+left+dx;
        let safe_index = if active { index } else { 0 };
        let value = image[noir_hw::index_pos::<8>(safe_index, R*C)];
        result[2+s] = if active { value } else { 0 };
    }
    result
}
