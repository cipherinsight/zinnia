// DS-1000 case 0508: literal translation of kernel.py (LITERAL_BASELINES.md).
// Coordinate arrays use prefix-count/equality compaction, then independent min/max scans.
// Dynamic loops run to the input shape with active flags; empty branches use safe dummy reads.
// @outputs: dynamic2 int[10x10]
global R: u32 = 10;
global C: u32 = 10;
fn main(A: [Field; R*C]) -> pub [Field; 2+R*C] {
    noir_hw::check_inputs(A);
    let mut keep = [false; 100]; let mut payload = [[0; 2]; 100];
    for i in 0..100 { keep[i] = A[i] != 0; payload[i] = [(i/10) as Field, (i%10) as Field]; }
    let (count, b) = nonzero(keep, payload);
    assert(count != 0); // B.min(0) raises if empty.
    let mut ys = [0; 100]; let mut xs = [0; 100];
    for i in 0..100 { ys[i] = b[i][0]; xs[i] = b[i][1]; }
    let top = reduce(ys, count, false); let left = reduce(xs, count, false);
    let bottom = reduce(ys, count, true); let right = reduce(xs, count, true);
    let bottom = bottom+1; let right = right+1;
    let present = true;
    crop(A, top, bottom, left, right, present)
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
