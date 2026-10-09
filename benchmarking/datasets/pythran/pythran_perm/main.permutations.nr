// pythran_perm__permutations (Pythran perm), Noir baseline.
// Kernel: every permutation of `iterable: Int[N]` (itertools' `permutations`
// with r = n), as a list of COUNT = N! tuples, in lexicographic order of the
// positions.
// Data movement only: the positions are constants of the circuit (computed
// from loop counters at compile time), so each output is a copy of an input
// element.
// @outputs: int[COUNT * N]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 4; // = iterable_0
global COUNT: u32 = 24; // = factorial(iterable_0)

// The k-th permutation of 0..N in lexicographic order: the digits of k in
// the factorial number system pick, position by position, among the
// positions not used yet.
fn permutation(k: u32) -> [u32; N] {
    let mut used: [bool; N] = [false; N];
    let mut order: [u32; N] = [0; N];
    let mut rest = k;
    let mut block = COUNT;
    for i in 0..N {
        block /= N - i;
        let digit = rest / block;
        rest %= block;
        let mut seen: u32 = 0;
        let mut pick: u32 = 0;
        for p in 0..N {
            if !used[p] {
                if seen == digit {
                    pick = p;
                }
                seen += 1;
            }
        }
        order[i] = pick;
        used[pick] = true;
    }
    order
}

fn main(iterable: [Field; N]) -> pub [Field; COUNT * N] {
    // Input domain: every input element is an int64.
    check_inputs(iterable);
    let mut result: [Field; COUNT * N] = [0; COUNT * N];
    for k in 0..COUNT {
        let order = permutation(k);
        for i in 0..N {
            result[k * N + i] = iterable[order[i]];
        }
    }
    result
}
