// pythran_perm__permutations: literal translation of kernel.py (LITERAL_BASELINES.md).
// The source cycles algorithm uses shape-only counters; Noir bounds while by N! including its terminal iteration.
// @outputs: int[COUNT * N]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 4; // = iterable_0
global COUNT: u32 = 24; // = factorial(iterable_0)

fn main(iterable: [Field; N]) -> pub [Field; COUNT * N] {
    // Input domain: every input element is an int64.
    check_inputs(iterable);
    let pool = iterable;
    let mut indices = [0 as u32; N];
    let mut cycles = [0 as u32; N];
    for i in 0..N { indices[i] = i; cycles[i] = N-i; }
    let mut result = [0; COUNT*N];
    for i in 0..N { result[i] = pool[indices[i]]; }
    // COUNT iterations include the final terminating iteration. All flags and indices are shape constants.
    let mut count: u32 = 1;
    let mut active = true;
    for _ in 0..COUNT {
        let mut broke = false;
        for t in 0..N {
            let i = N-1-t;
            if active & !broke {
                cycles[i] -= 1;
                if cycles[i] == 0 {
                    let first = indices[i];
                    for j in i..N-1 { indices[j] = indices[j+1]; }
                    indices[N-1] = first;
                    cycles[i] = N-i;
                } else {
                    let j = cycles[i];
                    let first = indices[i];
                    indices[i] = indices[N-j];
                    indices[N-j] = first;
                    for k in 0..N { result[count*N+k] = pool[indices[k]]; }
                    count += 1;
                    broke = true;
                }
            }
        }
        active = active & broke;
    }
    result
}
