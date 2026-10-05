// DS-1000 case 0402 -- Noir baseline. np.ravel_multi_index(index, dims, order='F').
// Fortran-order: idx = index[0] + dims[0]*index[1] + dims[0]*dims[1]*index[2] + ...
// Per numpy 'raise' semantics, every `index[k]` must satisfy
// `0 <= index[k] < dims[k]`; the per-axis discharge mirrors what the
// Zinnia-emitted variant inserts in `np_ravel_multi_index`.

use noir_fpc::{qge, qlt};

global RANK: u32 = 5;

fn main(dims: [Field; 5], a: [Field; 36864], index: [Field; 5], claimed: Field) -> pub Field {
    let _ = a[0];
    let _ = claimed;
    for k in 0..RANK {
        assert(qge(index[k], 0), "0402: index[k] must be >= 0");
        assert(qlt(index[k], dims[k]), "0402: index[k] must be < dims[k]");
    }
    let mut acc: Field = index[0];
    let mut stride: Field = dims[0];
    for k in 1..RANK {
        acc = acc + stride * index[k];
        stride = stride * dims[k];
    }
    acc
}
