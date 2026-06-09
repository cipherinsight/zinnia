// DS-1000 case 0441 -- Noir baseline. (number in a) on Integer[4].

global N: u32 = 4;

fn main(a: [Field; 4], number: Field) -> pub Field {
    let mut found: Field = 0;
    for k in 0..N {
        let eq: Field = if a[k] == number { 1 } else { 0 };
        // OR: found = found + eq - found*eq.
        found = found + eq - found * eq;
    }
    found
}
