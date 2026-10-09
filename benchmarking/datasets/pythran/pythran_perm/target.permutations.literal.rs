// pythran_perm__permutations: literal translation of kernel.py (LITERAL_BASELINES.md).
// The source cycles algorithm uses shape-only counters; Noir bounds while by N! including its terminal iteration.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 4; // = iterable_0
const COUNT: usize = 24; // = factorial(iterable_0)

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let iterable = inputs.ints("iterable", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let pool = q.load_ints(ctx, &iterable);
    let mut indices: Vec<usize> = (0..N).collect();
    let mut cycles: Vec<usize> = (1..=N).rev().collect();
    let mut result: Vec<Cell> = indices.iter().map(|i| pool[*i]).collect();
    loop {
        let mut broke = false;
        for i in (0..N).rev() {
            cycles[i] -= 1;
            if cycles[i] == 0 {
                let first = indices[i];
                for j in i..N-1 { indices[j] = indices[j+1]; }
                indices[N-1] = first;
                cycles[i] = N-i;
            } else {
                let j = cycles[i];
                indices.swap(i,N-j);
                for index in &indices { result.push(pool[*index]); }
                broke = true; break;
            }
        }
        if !broke { break; }
    }
    assert_eq!(result.len(),COUNT*N);
    out.ints(&result); Ok(())
}
