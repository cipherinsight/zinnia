// DS-1000 case 478 — SP1 guest program.
// kernel.py:
//     result = np.einsum('ii->i', a)   # diagonal of 4×4
//     save = result.copy()
//     a[...] = 0                        # zero the whole matrix
//     result[...] = save                # restore (no-op semantically)
//     return result
// Final result is the diagonal of a (length 4).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [[i64; 4]; 4] = [[0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = sp1_zkvm::io::read::<i64>();
        }
    }
    let mut claimed: [i64; 4] = [0; 4];
    for k in 0..4 {
        claimed[k] = sp1_zkvm::io::read::<i64>();
    }

    let mut result: [i64; 4] = [0; 4];
    for i in 0..4 {
        result[i] = a[i][i];
    }
    let save: [i64; 4] = result;
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = 0;
        }
    }
    for i in 0..4 {
        result[i] = save[i];
    }

    for k in 0..4 {
        assert_eq!(result[k], claimed[k]);
    }
}
