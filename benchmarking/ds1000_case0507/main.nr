// DS-1000 case 0507 -- Noir baseline (semantic translation).
// Python:
//   mask = im == 0
//   rows = np.flatnonzero((~mask).sum(axis=1))
//   cols = np.flatnonzero((~mask).sum(axis=0))
//   if rows.shape[0] == 0:
//       result = np.array([])
//   else:
//       result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//
// Step-by-step Noir translation: mask -> nmask -> row_sum/col_sum ->
// flatnonzero (upper-bounded packed array + count) -> min/max scan -> crop.

global R: u32 = 5;
global C: u32 = 6;

fn main(im: [Field; 30]) -> pub [Field; 36] {
    // Step 1: mask[i,j] = (im[i,j] == 0); nmask = ~mask.
    let mut mask: [u32; 30] = [0; 30];
    for i in 0..R {
        for j in 0..C {
            mask[i * C + j] = if im[i * C + j] == 0 { 1 } else { 0 };
        }
    }
    let mut nmask: [u32; 30] = [0; 30];
    for k in 0..30 {
        nmask[k] = 1 - mask[k];
    }

    // Step 2: per-row / per-column sums.
    let mut row_sum: [u32; 5] = [0; 5];
    for i in 0..R {
        let mut s: u32 = 0;
        for j in 0..C { s = s + nmask[i * C + j]; }
        row_sum[i] = s;
    }
    let mut col_sum: [u32; 6] = [0; 6];
    for j in 0..C {
        let mut s: u32 = 0;
        for i in 0..R { s = s + nmask[i * C + j]; }
        col_sum[j] = s;
    }

    // Step 3: np.flatnonzero(row_sum), np.flatnonzero(col_sum).
    let mut rows: [u32; 5] = [0; 5];
    let mut row_count: u32 = 0;
    for i in 0..R {
        if row_sum[i] != 0 {
            rows[row_count] = i;
            row_count = row_count + 1;
        }
    }
    let mut cols: [u32; 6] = [0; 6];
    let mut col_count: u32 = 0;
    for j in 0..C {
        if col_sum[j] != 0 {
            cols[col_count] = j;
            col_count = col_count + 1;
        }
    }

    // Step 4: rows.min()/max(), cols.min()/max() over live prefix.
    let mut rmin: u32 = R;
    let mut rmax: u32 = 0;
    for k in 0..R {
        if k < row_count {
            let v = rows[k];
            if v < rmin { rmin = v; }
            if v > rmax { rmax = v; }
        }
    }
    let mut cmin: u32 = C;
    let mut cmax: u32 = 0;
    for k in 0..C {
        if k < col_count {
            let v = cols[k];
            if v < cmin { cmin = v; }
            if v > cmax { cmax = v; }
        }
    }

    // Step 5: crop into the fixed output buffer + meta tail.
    let mut out: [Field; 36] = [0; 36];
    if row_count != 0 {
        for i in 0..R {
            for j in 0..C {
                let in_box = (i >= rmin) & (i <= rmax) & (j >= cmin) & (j <= cmax);
                out[i * C + j] = if in_box { im[i * C + j] } else { 0 };
            }
        }
        out[30] = rmin as Field;
        out[31] = rmax as Field;
        out[32] = cmin as Field;
        out[33] = cmax as Field;
        out[34] = (rmax - rmin + 1) as Field;
        out[35] = (cmax - cmin + 1) as Field;
    }
    out
}
