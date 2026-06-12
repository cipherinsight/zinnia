// DS-1000 case 0508 -- Noir baseline. Field convention (CONVENTIONS.md sec 3:
// Field inputs; u32 view permitted for internal index state).
// Mirrors Zinnia's lowering ALGORITHM step for step:
//   1. per-cell nonzero indicators (u32 0/1)
//   2. per-row / per-column nonzero COUNT chains, has-flag = (0 < count)
//   3. rmin/cmin: first-hit sentinel select-chain scan
//      rmax/cmax: last-hit select-chain scan
//   4. in-crop masks from (origin + offset) comparisons
//   5. origin one-hot selectors, factored two-stage shift gather with
//      per-shift one-hot membership asserts
//   6. packed output + [h*w, h, w] runtime-shape metadata (numpy contract)

global R: u32 = 10;
global C: u32 = 10;
global SENTINEL: u32 = 10;

fn main(a: [Field; 100]) -> pub [Field; 103] {
    // 1. Per-cell nonzero indicators.
    let mut nz: [u32; 100] = [0; 100];
    for k in 0..100 {
        nz[k] = if a[k] == 0 { 0 } else { 1 };
    }
    // 2. Count chains + has flags.
    let mut row_has: [u32; 10] = [0; 10];
    for i in 0..R {
        let mut cnt: u32 = 0;
        for j in 0..C {
            cnt = cnt + nz[i * C + j];
        }
        row_has[i] = if 0 < cnt { 1 } else { 0 };
    }
    let mut col_has: [u32; 10] = [0; 10];
    for j in 0..C {
        let mut cnt: u32 = 0;
        for i in 0..R {
            cnt = cnt + nz[i * C + j];
        }
        col_has[j] = if 0 < cnt { 1 } else { 0 };
    }
    // 3. First-hit sentinel scans (select-chain form) and last-hit scans.
    let mut rmin: u32 = row_has[0] * 0 + (1 - row_has[0]) * SENTINEL;
    for i in 1..R {
        let not_found: u32 = if rmin == SENTINEL { 1 } else { 0 };
        let take: u32 = not_found * row_has[i];
        rmin = take * i + (1 - take) * rmin;
    }
    let mut cmin: u32 = col_has[0] * 0 + (1 - col_has[0]) * SENTINEL;
    for j in 1..C {
        let not_found: u32 = if cmin == SENTINEL { 1 } else { 0 };
        let take: u32 = not_found * col_has[j];
        cmin = take * j + (1 - take) * cmin;
    }
    let mut rmax: u32 = row_has[1] * 1;
    for i in 2..R {
        rmax = row_has[i] * i + (1 - row_has[i]) * rmax;
    }
    let mut cmax: u32 = col_has[1] * 1;
    for j in 2..C {
        cmax = col_has[j] * j + (1 - col_has[j]) * cmax;
    }
    let rstop: u32 = rmax + 1;
    let cstop: u32 = cmax + 1;

    // 5. Origin one-hot selectors.
    let mut rowsel: [Field; 10] = [0; 10];
    for r in 0..R {
        rowsel[r] = if rmin == r { 1 } else { 0 };
    }
    let mut colsel: [Field; 10] = [0; 10];
    for c in 0..C {
        colsel[c] = if cmin == c { 1 } else { 0 };
    }
    // Per-shift one-hot membership asserts: the truncated selector family
    // for shift s is completed by the boundary indicator (origin beyond
    // R-s implies the shifted read would leave the grid; the in-crop mask
    // is 0 there, but the family must still be one-hot for the gather).
    for s in 1..R {
        let mut total: Field = 0;
        for r in 0..R {
            if r + s < R {
                total = total + rowsel[r];
            }
        }
        let boundary: Field = if rmin >= R - s { 1 } else { 0 };
        assert(total + boundary == 1);
    }
    for s in 1..C {
        let mut total: Field = 0;
        for c in 0..C {
            if c + s < C {
                total = total + colsel[c];
            }
        }
        let boundary: Field = if cmin >= C - s { 1 } else { 0 };
        assert(total + boundary == 1);
    }

    // 4+5. Packed shift gather, masked by in-crop comparisons on
    // (origin + offset) exactly as the auto-lowering does.
    let mut out: [Field; 103] = [0; 103];
    for i in 0..R {
        let ri: u32 = rmin + i;
        let row_ok: u32 = if ri < R { 1 } else { 0 };
        let row_in: u32 = if ri < rstop { 1 } else { 0 };
        let rmask: u32 = row_ok * row_in;
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
            let cj: u32 = cmin + j;
            let col_ok: u32 = if cj < C { 1 } else { 0 };
            let col_in: u32 = if cj < cstop { 1 } else { 0 };
            let cmask: u32 = col_ok * col_in;
            let mut v: Field = 0;
            for c in 0..C {
                if c + j < C {
                    v = v + colsel[c] * t[c + j];
                }
            }
            out[3 + i * C + j] = ((rmask * cmask) as Field) * v;
        }
    }
    // 6. Runtime-shape metadata (clamped diffs, select form).
    let h_raw: u32 = rstop - rmin;
    let h_neg: u32 = if h_raw < 0 { 1 } else { 0 };
    let h: u32 = h_neg * 0 + (1 - h_neg) * h_raw;
    let w_raw: u32 = cstop - cmin;
    let w_neg: u32 = if w_raw < 0 { 1 } else { 0 };
    let w: u32 = w_neg * 0 + (1 - w_neg) * w_raw;
    out[0] = (h * w) as Field;
    out[1] = h as Field;
    out[2] = w as Field;
    out
}
