// DS-1000 case 0508 -- Noir baseline. Field convention (CONVENTIONS.md sec 3:
// Field inputs; u32 view permitted for internal index state).
// Mirrors Zinnia's argwhere_axis0_bbox_bounds lowering: per-row/per-column
// "any nonzero" indicators, then linear scan for first/last hit.

global R: u32 = 10;
global C: u32 = 10;

fn main(a: [Field; 100]) -> pub [Field; 106] {
    let mut row_has: [u32; 10] = [0; 10];
    for i in 0..R {
        let mut any: u32 = 0;
        for j in 0..C {
            if a[i * C + j] != 0 { any = 1; }
        }
        row_has[i] = any;
    }
    let mut col_has: [u32; 10] = [0; 10];
    for j in 0..C {
        let mut any: u32 = 0;
        for i in 0..R {
            if a[i * C + j] != 0 { any = 1; }
        }
        col_has[j] = any;
    }
    let mut rmin: u32 = R - 1;
    let mut found: u32 = 0;
    for i in 0..R {
        if (found == 0) & (row_has[i] == 1) { rmin = i; found = 1; }
    }
    let mut rmax: u32 = 0;
    for i in 0..R { if row_has[i] == 1 { rmax = i; } }
    let mut cmin: u32 = C - 1;
    let mut cfound: u32 = 0;
    for j in 0..C {
        if (cfound == 0) & (col_has[j] == 1) { cmin = j; cfound = 1; }
    }
    let mut cmax: u32 = 0;
    for j in 0..C { if col_has[j] == 1 { cmax = j; } }
    let mut out: [Field; 106] = [0; 106];
    for i in 0..R {
        for j in 0..C {
            let in_box = (i >= rmin) & (i <= rmax) & (j >= cmin) & (j <= cmax);
            out[i * C + j] = if in_box { a[i * C + j] } else { 0 };
        }
    }
    out[100] = rmin as Field;
    out[101] = rmax as Field;
    out[102] = cmin as Field;
    out[103] = cmax as Field;
    out[104] = (rmax - rmin + 1) as Field;
    out[105] = (cmax - cmin + 1) as Field;
    out
}
