// DS-1000 case 0500 -- Noir baseline. a.reshape(-1, 3) on Float[54] -> (18, 3).
// Pure reshape; values pass through.

fn main(a: [Field; 54]) -> pub [Field; 54] {
    a
}
