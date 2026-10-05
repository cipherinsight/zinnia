// DS-1000 case 0317, Noir baseline.
// Kernel: `np.array(a)` on Float[10, 20, 5]: a copy.
// @outputs: float[1000]

fn main(a: [Field; 1000]) -> pub [Field; 1000] {
    a
}
