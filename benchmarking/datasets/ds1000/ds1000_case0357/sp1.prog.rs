// DS-1000 case 357 — SP1 guest program.
// Kernel:
//   sort_indices = np.argsort(a, axis=0)[::-1, :, :]
//   c = b[sort_indices, static_indices[1], static_indices[2]]
// Equivalent to argsort in descending order along axis 0.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a = [[[0.0f64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { a[i][j][k] = sp1_zkvm::io::read::<f64>(); } } }
    let mut b = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { b[i][j][k] = sp1_zkvm::io::read::<i64>(); } } }
    let mut claimed = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { claimed[i][j][k] = sp1_zkvm::io::read::<i64>(); } } }

    // ascending argsort along axis 0
    let mut asort = [[[0usize; 3]; 3]; 3];
    for j in 0..3 {
        for k in 0..3 {
            let mut idx = [0usize, 1, 2];
            if a[idx[0]][j][k] > a[idx[1]][j][k] { idx.swap(0, 1); }
            if a[idx[1]][j][k] > a[idx[2]][j][k] { idx.swap(1, 2); }
            if a[idx[0]][j][k] > a[idx[1]][j][k] { idx.swap(0, 1); }
            for i in 0..3 { asort[i][j][k] = idx[i]; }
        }
    }
    // reverse axis 0
    let mut sort_indices = [[[0usize; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                sort_indices[i][j][k] = asort[2 - i][j][k];
            }
        }
    }
    let mut c = [[[0i64; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        c[i][j][k] = b[sort_indices[i][j][k]][j][k];
    } } }
    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        assert_eq!(c[i][j][k], claimed[i][j][k]);
    } } }
}
