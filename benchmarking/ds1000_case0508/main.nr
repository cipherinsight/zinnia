// DS-1000 case 0508 -- Noir baseline. Field convention (CONVENTIONS.md sec 3:
// Field inputs; u32 view permitted for internal index state).
// Binds the SAME contract as Zinnia: the numpy result `A[ystart:ystop,
// xstart:xstop]` packed to the origin of a 10x10 envelope, preceded by its
// runtime shape metadata [h*w, h, w]. Cells outside the runtime (h, w)
// rectangle are zero. This requires a genuine 2-D dynamic shift gather
// (factored row stage then column stage), not just in-place masking.

global R: u32 = 10;
global C: u32 = 10;

fn main(a: [Field; 100]) -> pub [Field; 103] {
    // Per-row / per-column "any nonzero" indicators.
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
    // Bounding box: first/last set row and column. Inputs guarantee at
    // least one nonzero entry (np.argwhere(...).min() errors otherwise).
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

    let h: u32 = rmax + 1 - rmin;
    let w: u32 = cmax + 1 - cmin;

    // One-hot selectors over the crop origin.
    let mut rowsel: [Field; 10] = [0; 10];
    for r in 0..R {
        rowsel[r] = if rmin == r { 1 } else { 0 };
    }
    let mut colsel: [Field; 10] = [0; 10];
    for c in 0..C {
        colsel[c] = if cmin == c { 1 } else { 0 };
    }

    // Packed shift gather: out[i][j] = a[rmin+i][cmin+j] when (i, j) is
    // inside the runtime rectangle, else 0. Factored: a row stage shared
    // across the 10 columns of each output row, then a column stage.
    // Source coordinates beyond the grid are skipped -- whenever the
    // in-crop mask is 1, rmin+i <= rmax <= 9 (resp. cmin+j <= 9), so the
    // truncated sums are complete exactly when their value is used.
    let mut out: [Field; 103] = [0; 103];
    out[0] = (h * w) as Field;
    out[1] = h as Field;
    out[2] = w as Field;
    for i in 0..R {
        let mut t: [Field; 10] = [0; 10];
        for k in 0..C {
            let mut acc: Field = 0;
            for r in 0..R {
                if r + i < R {
                    acc = acc + rowsel[r] * a[(r + i) * C + k];
                }
            }
            t[k] = acc;
        }
        for j in 0..C {
            let mut v: Field = 0;
            for c in 0..C {
                if c + j < C {
                    v = v + colsel[c] * t[c + j];
                }
            }
            let in_crop: bool = (i < h) & (j < w);
            out[3 + i * C + j] = if in_crop { v } else { 0 };
        }
    }
    out
}
