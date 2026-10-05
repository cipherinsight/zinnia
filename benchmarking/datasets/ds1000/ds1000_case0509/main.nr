// DS-1000 case 0509 -- Noir baseline (semantic translation).
// Python:
//   mask = im == 0
//   rows = np.flatnonzero((mask).sum(axis=1))
//   cols = np.flatnonzero((mask).sum(axis=0))
//   if rows.shape[0] == 0:
//       result = np.array([])
//   else:
//       result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//
// Step-by-step Noir translation:
//   1. mask[i,j] = (im[i,j] == 0).
//   2. row_sum[i] = sum_j mask[i,j];   col_sum[j] = sum_i mask[i,j].
//   3. rows[0..row_count] = np.flatnonzero(row_sum)  (upper-bounded packed array).
//      cols[0..col_count] = np.flatnonzero(col_sum).
//   4. rmin = rows[0..row_count].min(), rmax = rows[0..row_count].max()
//      (linear scan over the live prefix).  Similarly for cmin/cmax.
//   5. Crop im[rmin..rmax+1, cmin..cmax+1]; pack into a fixed-size buffer
//      with rmin/rmax/cmin/cmax/h/w meta in the last 6 slots.

global R: u32 = 10;
global C: u32 = 10;

fn main(im: [Field; 100]) -> pub [Field; 106] {
    // Step 1: build the mask.
    let mut mask: [u32; 100] = [0; 100];
    for i in 0..R {
        for j in 0..C {
            mask[i * C + j] = if im[i * C + j] == 0 { 1 } else { 0 };
        }
    }

    // Step 2: per-row / per-column sums of the mask.
    let mut row_sum: [u32; 10] = [0; 10];
    for i in 0..R {
        let mut s: u32 = 0;
        for j in 0..C {
            s = s + mask[i * C + j];
        }
        row_sum[i] = s;
    }
    let mut col_sum: [u32; 10] = [0; 10];
    for j in 0..C {
        let mut s: u32 = 0;
        for i in 0..R {
            s = s + mask[i * C + j];
        }
        col_sum[j] = s;
    }

    // Step 3: np.flatnonzero(row_sum), np.flatnonzero(col_sum)
    // as upper-bounded packed arrays.
    let mut rows: [u32; 10] = [0; 10];
    let mut row_count: u32 = 0;
    for i in 0..R {
        if row_sum[i] != 0 {
            rows[row_count] = i;
            row_count = row_count + 1;
        }
    }
    let mut cols: [u32; 10] = [0; 10];
    let mut col_count: u32 = 0;
    for j in 0..C {
        if col_sum[j] != 0 {
            cols[col_count] = j;
            col_count = col_count + 1;
        }
    }

    // Step 4: rmin = rows.min(), rmax = rows.max(), ditto cols, by linear
    // scan over the [0, *_count) prefix. (Inactive slots are zero-padded;
    // we initialize min to a sentinel above R and skip past row_count.)
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

    // Step 5: crop. Output buffer: 100 image cells + 6 meta values.
    // Empty (row_count == 0) case: zeros and h=w=0.
    let mut out: [Field; 106] = [0; 106];
    if row_count != 0 {
        for i in 0..R {
            for j in 0..C {
                let in_box = (i >= rmin) & (i <= rmax) & (j >= cmin) & (j <= cmax);
                out[i * C + j] = if in_box { im[i * C + j] } else { 0 };
            }
        }
        out[100] = rmin as Field;
        out[101] = rmax as Field;
        out[102] = cmin as Field;
        out[103] = cmax as Field;
        out[104] = (rmax - rmin + 1) as Field;
        out[105] = (cmax - cmin + 1) as Field;
    }
    out
}
