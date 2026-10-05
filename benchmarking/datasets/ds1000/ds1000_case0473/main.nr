// DS-1000 case 0473 -- Noir baseline.
// any(np.array_equal(c, x) for x in CNTS) -- always False (shape mismatch).

fn main(c: [Field; 8], cnts: [Field; 3]) -> pub Field {
    let _ = c[0];
    let _ = cnts[0];
    0
}
