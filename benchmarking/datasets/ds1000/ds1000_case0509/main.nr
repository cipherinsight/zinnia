// DS-1000 case 0509, Noir baseline.
// Kernel: on im: Float[10, 10], `mask = im == 0`; crop `im` to the rows and
// columns spanned by those containing a 0
// (`im[rows.min():rows.max()+1, cols.min():cols.max()+1]`), or `np.array([])`
// when no entry is 0.
// Output: public (rows, cols), then the crop compact and row-major,
// zero-padded to 10 x 10; the empty result is (0, 0).
// Construction (each test made once, `==` exact as in the kernel):
// - a row contains a 0 iff the product of its entries is 0 (exact for any
//   field elements: a field has no zero divisors), one zero test per row and
//   per column;
// - lead_i = "rows 0..=i contain no 0" and after_i = "rows i.. contain no 0"
//   are running products of the complements of those bits; the first row with
//   a 0 is the one-hot top_d = lead_{d-1} - lead_d, and the height is
//   10 - sum(lead) - sum(after) + 10 lead_9 (0 when no entry is 0); the same
//   for columns;
// - out[u][v] = im[top + u][left + v] is a one-hot shift of the rows, then of
//   the columns. Rows and columns outside the span need not be zero here, so
//   the selector of output row u reading source row d + u is
//   top_d * (1 - after_{d+u}), which is 0 past the span (likewise for columns);
// - the crop, now at the top-left of a zero grid, is laid out with stride
//   `width`: one fixed reshuffle per possible width, selected by the one-hot
//   of `width`.
// Native indexing would need u32 arithmetic on the data-dependent offsets,
// whose range checks cost more here than these one-hot sums.
// @outputs: dynamic2 float[10x10]

global R: u32 = 10;
global C: u32 = 10;

fn main(im: [Field; R * C]) -> pub [Field; 2 + R * C] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(im);
    let mut row_hit: [Field; R] = [0; R];
    for i in 0..R {
        let mut prod: Field = 1;
        for j in 0..C {
            prod *= im[i * C + j];
        }
        row_hit[i] = (prod == 0) as Field;
    }
    let mut col_hit: [Field; C] = [0; C];
    for j in 0..C {
        let mut prod: Field = 1;
        for i in 0..R {
            prod *= im[i * C + j];
        }
        col_hit[j] = (prod == 0) as Field;
    }
    let (top, row_after, height) = extent(row_hit);
    let (left, col_after, width) = extent(col_hit);

    let grid = shift(im, span_selectors(top, row_after), span_selectors(left, col_after));
    let crop = compact(grid, width);
    let mut out: [Field; 2 + R * C] = [0; 2 + R * C];
    out[0] = height;
    out[1] = width;
    for s in 0..R * C {
        out[2 + s] = crop[s];
    }
    out
}

// From hit[i] = 1 when line i is selected: the one-hot of the first selected
// line, after[i] = 1 when line i comes after the last selected one, and the
// number of lines from the first to the last selected one (0 when none is).
fn extent<let N: u32>(hit: [Field; N]) -> ([Field; N], [Field; N], Field) {
    let mut lead: [Field; N] = [0; N]; // lead[i]: no line in 0..=i selected
    lead[0] = 1 - hit[0];
    for i in 1..N {
        lead[i] = lead[i - 1] * (1 - hit[i]);
    }
    let mut after: [Field; N] = [0; N]; // after[i]: no line in i.. selected
    after[N - 1] = 1 - hit[N - 1];
    for t in 1..N {
        after[N - 1 - t] = after[N - t] * (1 - hit[N - 1 - t]);
    }
    let mut first: [Field; N] = [0; N];
    first[0] = hit[0];
    for d in 1..N {
        first[d] = lead[d - 1] - lead[d];
    }
    // N - sum(lead) - sum(after) + N lead[N-1]
    let mut len: Field = (N as Field) * (1 + lead[N - 1]);
    for i in 0..N {
        len -= lead[i] + after[i];
    }
    (first, after, len)
}

// sel[u][d] = first[d] * (1 - after[d + u]): output line u reads source line
// d + u only while that line is inside the span. For u = 0 it is first[d]
// (the first selected line is inside the span).
fn span_selectors<let N: u32>(first: [Field; N], after: [Field; N]) -> [[Field; N]; N] {
    let mut sel: [[Field; N]; N] = [[0; N]; N];
    sel[0] = first;
    for u in 1..N {
        for d in 0..N - u {
            sel[u][d] = first[d] * (1 - after[d + u]);
        }
    }
    sel
}

// out[u][v] = sum over d, e of row_sel[u][d] * col_sel[v][e] * src[u + d][v + e]
// (positions past the end of src contribute nothing): a shift of the rows,
// then of the columns.
fn shift(src: [Field; R * C], row_sel: [[Field; R]; R], col_sel: [[Field; C]; C]) -> [Field; R * C] {
    let mut rows: [Field; R * C] = [0; R * C];
    for u in 0..R {
        for j in 0..C {
            let mut x: Field = 0;
            for d in 0..R - u {
                x += row_sel[u][d] * src[(u + d) * C + j];
            }
            rows[u * C + j] = x;
        }
    }
    let mut out: [Field; R * C] = [0; R * C];
    for u in 0..R {
        for v in 0..C {
            let mut x: Field = 0;
            for e in 0..C - v {
                x += col_sel[v][e] * rows[u * C + v + e];
            }
            out[u * C + v] = x;
        }
    }
    out
}

// The crop held at the top-left of the R x C grid (zero elsewhere), laid out
// compact and row-major for its width w: slot s holds grid[s / w][s % w]. One
// fixed reshuffle per possible width, selected by the one-hot [w == W].
fn compact(grid: [Field; R * C], width: Field) -> [Field; R * C] {
    let mut is_width: [Field; C] = [0; C];
    for k in 0..C {
        is_width[k] = (width == (k + 1) as Field) as Field;
    }
    let mut out: [Field; R * C] = [0; R * C];
    for s in 0..R * C {
        let mut x: Field = 0;
        for k in 0..C {
            let w = k + 1;
            if s / w < R {
                x += is_width[k] * grid[(s / w) * C + s % w];
            }
        }
        out[s] = x;
    }
    out
}
