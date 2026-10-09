//! npbench_polybench_gramschmidt_gramschmidt__kernel (NPBench polybench/gramschmidt), halo2-base baseline.
//! Kernel: modified Gram-Schmidt on A: Float[NR, NC]; the output is the tuple
//! `(Q, R)`, Q: [NR, NC], R: [NC, NC]. For each column `k`:
//! `R[k, k] = sqrt(np.dot(A[:, k], A[:, k]))`, `Q[:, k] = A[:, k] / R[k, k]`,
//! and for each later column `j`: `R[k, j] = np.dot(Q[:, k], A[:, j])`,
//! `A[:, j] -= Q[:, k] * R[k, j]`.
//! Each dot product is one reduction, rescaled once; the root is the exact
//! floor square root; the quotients divide by the root, positive (a zero
//! column has no proof: NumPy gives nan); the products of the update are
//! rounded one by one. `R` starts as zeros and only its upper triangle is
//! written: the lower triangle is a constant output, not public
//! (CONVENTIONS 1.3).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NR: usize = 14; // = A_0
const NC: usize = 12; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[NR, NC])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let mut columns: Vec<Vec<Cell>> = (0..NC).map(|j| (0..NR).map(|i| a[i * NC + j]).collect()).collect();
    let mut q_columns: Vec<Vec<Cell>> = Vec::with_capacity(NC);
    let mut r_rows: Vec<Vec<Cell>> = Vec::with_capacity(NC); // R[k, k:]
    for k in 0..NC {
        let norm_sq = q.dot(ctx, &columns[k], &columns[k]);
        let r_kk = q.sqrt(ctx, norm_sq.cell);
        let q_k: Vec<Cell> = columns[k].iter().map(|x| q.div_pos(ctx, *x, r_kk).cell).collect();
        let mut r_row = vec![r_kk.cell];
        for j in k + 1..NC {
            let r_kj = q.dot(ctx, &q_k, &columns[j]).cell;
            for i in 0..NR {
                let product = q.mul(ctx, q_k[i], r_kj);
                columns[j][i] = q.sub(ctx, columns[j][i], product.cell);
            }
            r_row.push(r_kj);
        }
        q_columns.push(q_k);
        r_rows.push(r_row);
    }
    let q_matrix: Vec<Cell> = (0..NR).flat_map(|i| q_columns.iter().map(move |column| column[i])).collect();
    out.floats(&q_matrix);
    for (k, row) in r_rows.iter().enumerate() {
        out.const_floats(&vec![0.0; k]);
        out.floats(row);
    }
    Ok(())
}
