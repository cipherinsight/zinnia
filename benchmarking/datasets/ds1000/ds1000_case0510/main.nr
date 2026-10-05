// DS-1000 case 0510, Noir baseline.
// Kernel: on im: Float[10, 10], `mask = im == 0`; crop `im` to the bounding
// box of its nonzero entries (`rows`, `cols` from `~mask`;
// `im[rows.min():rows.max()+1, cols.min():cols.max()+1]`), or `np.array([])`
// when every entry is 0.
// Output: public (rows, cols), then the crop compact and row-major,
// zero-padded to 10 x 10; the empty result is (0, 0).
// Construction (each test made once, `==` exact as in the kernel):
// - one zero test per entry (`x == 0`, exact for any field element; the inputs
//   carry no range proof); a row or column is all zero iff the product of its
//   entries' zero bits is 1;
// - lead_i = "rows 0..=i are all zero" and after_i = "rows i.. are all zero"
//   are running products of those bits; the first nonzero row is the one-hot
//   top_d = lead_{d-1} - lead_d, and the height is
//   10 - sum(lead) - sum(after) + 10 lead_9 (0 for an all-zero image); the
//   same for columns;
// - out[u][v] = im[top + u][left + v] is a one-hot shift of the rows, then of
//   the columns. Positions past the box read rows or columns outside the
//   bounding box, which are all zero, so the padding needs no mask;
// - the crop, now at the top-left of a zero grid, is laid out with stride
//   `width`: one fixed reshuffle per possible width, selected by the one-hot
//   of `width`.
// Native indexing would need u32 arithmetic on the data-dependent offsets,
// whose range checks cost more here than these one-hot sums.
// @outputs: dynamic2 float[10x10]

global R: u32 = 10;
global C: u32 = 10;

fn main(im: [Field; R * C]) -> pub [Field; 2 + R * C] {
    let mut zero: [Field; R * C] = [0; R * C];
    for k in 0..R * C {
        zero[k] = (im[k] == 0) as Field;
    }
    let mut row_zero: [Field; R] = [0; R];
    for i in 0..R {
        let mut all: Field = 1;
        for j in 0..C {
            all *= zero[i * C + j];
        }
        row_zero[i] = all;
    }
    let mut col_zero: [Field; C] = [0; C];
    for j in 0..C {
        let mut all: Field = 1;
        for i in 0..R {
            all *= zero[i * C + j];
        }
        col_zero[j] = all;
    }
    let (top, height) = extent(row_zero);
    let (left, width) = extent(col_zero);

    let grid = shift(im, top, left);
    let crop = compact(grid, width);
    let mut out: [Field; 2 + R * C] = [0; 2 + R * C];
    out[0] = height;
    out[1] = width;
    for s in 0..R * C {
        out[2 + s] = crop[s];
    }
    out
}

// From skip[i] = 1 when line i is all zero: the one-hot of the first line that
// is not, and the number of lines from it to the last one that is not (0 when
// every line is skipped).
fn extent<let N: u32>(skip: [Field; N]) -> ([Field; N], Field) {
    let mut lead: [Field; N] = [0; N]; // lead[i]: lines 0..=i skipped
    lead[0] = skip[0];
    for i in 1..N {
        lead[i] = lead[i - 1] * skip[i];
    }
    let mut after: [Field; N] = [0; N]; // after[i]: lines i.. skipped
    after[N - 1] = skip[N - 1];
    for t in 1..N {
        after[N - 1 - t] = after[N - t] * skip[N - 1 - t];
    }
    let mut first: [Field; N] = [0; N];
    first[0] = 1 - lead[0];
    for d in 1..N {
        first[d] = lead[d - 1] - lead[d];
    }
    // N - sum(lead) - sum(after) + N lead[N-1]
    let mut len: Field = (N as Field) * (1 + lead[N - 1]);
    for i in 0..N {
        len -= lead[i] + after[i];
    }
    (first, len)
}

// out[u][v] = src[top + u][left + v] for one-hot top and left (positions past
// the end of src read 0): a shift of the rows, then of the columns.
fn shift(src: [Field; R * C], top: [Field; R], left: [Field; C]) -> [Field; R * C] {
    let mut rows: [Field; R * C] = [0; R * C];
    for u in 0..R {
        for j in 0..C {
            let mut x: Field = 0;
            for d in 0..R - u {
                x += top[d] * src[(u + d) * C + j];
            }
            rows[u * C + j] = x;
        }
    }
    let mut out: [Field; R * C] = [0; R * C];
    for u in 0..R {
        for v in 0..C {
            let mut x: Field = 0;
            for e in 0..C - v {
                x += left[e] * rows[u * C + v + e];
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
