// DS-1000 case 0317 -- Noir baseline. result = np.array(a) on Float[10,20,5].
// Pure copy; Q32 fixed-point values pass through unchanged.

fn main(a: [Field; 1000]) -> pub [Field; 1000] {
    a
}
