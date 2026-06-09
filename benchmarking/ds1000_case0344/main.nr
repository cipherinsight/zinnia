// DS-1000 case 0344 -- Noir baseline.
// np.array(list(mystr), dtype=int) echoes the integer per the halo2 baseline.

fn main(mystr: Field) -> pub Field {
    mystr
}
