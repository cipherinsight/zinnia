// DS-1000 case 356 — SP1 guest program.
// Kernel:
//   sort_indices = np.argsort(a, axis=0)            # shape (3,3,3) of indices
//   static_indices = np.indices(a.shape)
//   c = b[sort_indices, static_indices[1], static_indices[2]]
// For each (j, k), argsort the column a[:, j, k] (3 elements) -> indices,
// then gather b[sort_indices[i, j, k], j, k] elementwise.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // a: 3x3x3 floats; b: 3x3x3 ints.
    let mut a = [[[0.0f64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { a[i][j][k] = sp1_zkvm::io::read::<f64>(); } } }
    let mut b = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { b[i][j][k] = sp1_zkvm::io::read::<i64>(); } } }
    let mut claimed = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { claimed[i][j][k] = sp1_zkvm::io::read::<i64>(); } } }

    // argsort along axis 0 for each (j, k).
    let mut sort_indices = [[[0usize; 3]; 3]; 3];
    for j in 0..3 {
        for k in 0..3 {
            // Stable insertion sort of [0,1,2] by a[idx][j][k] ascending.
            let mut idx = [0usize, 1, 2];
            // Sort by simple compare network (3 elements).
            if a[idx[0]][j][k] > a[idx[1]][j][k] { idx.swap(0, 1); }
            if a[idx[1]][j][k] > a[idx[2]][j][k] { idx.swap(1, 2); }
            if a[idx[0]][j][k] > a[idx[1]][j][k] { idx.swap(0, 1); }
            for i in 0..3 { sort_indices[i][j][k] = idx[i]; }
        }
    }

    let mut c = [[[0i64; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j][k] = b[sort_indices[i][j][k]][j][k];
            }
        }
    }

    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        assert_eq!(c[i][j][k], claimed[i][j][k]);
    } } }
}
