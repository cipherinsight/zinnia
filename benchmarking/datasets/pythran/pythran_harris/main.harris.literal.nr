// pythran_harris__harris: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compute full differences before slicing, including discarded boundary values.

// @outputs: float[(M - 1) * (N - 1)]

use noir_hw::{check_input, mul};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 4; // = I_0
global N: u32 = 6; // = I_1

global K_Q: Field = 214748365; // round(0.05 * 2^32)

fn main(I: [[Field; N]; M]) -> pub [[Field; N - 1]; M - 1] {
    for i in 0..M {
        for j in 0..N {
            check_input(I[i][j]);
        }
    }
    let mut horizontal = [[0; N]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N {
            horizontal[i][j] = I[i + 1][j] - I[i][j];
        }
    }
    let mut dx = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            dx[i][j] = horizontal[i][j + 1];
        }
    }
    let mut vertical = [[0; N - 1]; M];
    for i in 0..M {
        for j in 0..N - 1 {
            vertical[i][j] = I[i][j + 1] - I[i][j];
        }
    }
    let mut dy = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            dy[i][j] = vertical[i + 1][j];
        }
    }
    let mut a = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            a[i][j] = mul(dx[i][j], dx[i][j]);
        }
    }
    let mut b = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            b[i][j] = mul(dy[i][j], dy[i][j]);
        }
    }
    let mut c = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            c[i][j] = mul(dx[i][j], dy[i][j]);
        }
    }
    let mut tr = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            tr[i][j] = a[i][j] + b[i][j];
        }
    }
    let mut ab = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            ab[i][j] = mul(a[i][j], b[i][j]);
        }
    }
    let mut cc = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            cc[i][j] = mul(c[i][j], c[i][j]);
        }
    }
    let mut det = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            det[i][j] = ab[i][j] - cc[i][j];
        }
    }
    let mut ktr = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            ktr[i][j] = mul(K_Q, tr[i][j]);
        }
    }
    let mut ktrtr = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            ktrtr[i][j] = mul(ktr[i][j], tr[i][j]);
        }
    }
    let mut result = [[0; N - 1]; M - 1];
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            result[i][j] = det[i][j] - ktrtr[i][j];
        }
    }
    result
}
