// DS-1000 case 463 — SP1 guest program.
// kernel.py:
//   selection = np.ones((len(a), 1), dtype=bool)
//   selection[1:] = a[1:] != a[:-1]
//   selection &= a != 0
//   result = a[selection].reshape(-1, 1)
// Removes adjacent (before removing) duplicates and any zero values.
// Input shape: 20×1 i64. We pass the claimed compact result as a fixed-length
// padded buffer of length 20 plus its actual length, then recompute and assert.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 20;
    let mut a: [i64; N] = [0; N];
    for k in 0..N {
        a[k] = sp1_zkvm::io::read::<i64>();
    }
    let claimed_len: u64 = sp1_zkvm::io::read::<u64>();
    let mut claimed: [i64; N] = [0; N];
    for k in 0..N {
        claimed[k] = sp1_zkvm::io::read::<i64>();
    }

    // selection[0] = true; selection[k] = (a[k] != a[k-1]) for k>=1.
    let mut selection: [bool; N] = [false; N];
    selection[0] = true;
    for k in 1..N {
        selection[k] = a[k] != a[k - 1];
    }
    // selection &= (a != 0)
    for k in 0..N {
        selection[k] = selection[k] && (a[k] != 0);
    }

    // Gather the selected values into a compact buffer.
    let mut out: [i64; N] = [0; N];
    let mut out_len: u64 = 0;
    for k in 0..N {
        if selection[k] {
            out[out_len as usize] = a[k];
            out_len += 1;
        }
    }

    assert_eq!(out_len, claimed_len);
    for k in 0..(claimed_len as usize) {
        assert_eq!(out[k], claimed[k]);
    }
}
