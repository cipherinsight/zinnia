// DS-1000 case 0367 -- Noir baseline. copy.deepcopy(array_of_arrays) on Integer[3].
// pairs is loaded but unused (Zinnia DCE).

fn main(pairs: [Field; 6], array_of_arrays: [Field; 3]) -> pub [Field; 3] {
    let _ = pairs[0];
    array_of_arrays
}
