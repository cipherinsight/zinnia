//! pythran_perm__permutations (Pythran perm), halo2-base baseline.
//! Kernel: every permutation of `iterable: Int[N]` (itertools'
//! `permutations` with `r = n`), as a list of `COUNT = N!` tuples, in
//! lexicographic order of the positions.
//! Data movement only: the positions are constants of the circuit, so each
//! output is a copy of a loaded input element (public cells, no gates).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 4; // = iterable_0
const COUNT: usize = 24; // = factorial(iterable_0)

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let iterable = inputs.ints("iterable", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let pool = q.load_ints(ctx, &iterable);
    let orders = permutations();
    assert_eq!(orders.len(), COUNT);
    let result: Vec<Cell> = orders.iter().flat_map(|order| order.iter().map(|&i| pool[i])).collect();
    out.ints(&result);
    Ok(())
}

/// The permutations of `0..N` in lexicographic order (each step is the
/// standard next permutation: the longest non-increasing suffix is reversed
/// after its predecessor is swapped with the next larger position in it).
fn permutations() -> Vec<[usize; N]> {
    let mut order: [usize; N] = std::array::from_fn(|i| i);
    let mut all = vec![order];
    loop {
        let Some(pivot) = (1..N).rev().find(|&i| order[i - 1] < order[i]).map(|i| i - 1) else {
            return all;
        };
        let successor = (pivot + 1..N).rev().find(|&j| order[j] > order[pivot]).expect("suffix has a larger element");
        order.swap(pivot, successor);
        order[pivot + 1..].reverse();
        all.push(order);
    }
}
