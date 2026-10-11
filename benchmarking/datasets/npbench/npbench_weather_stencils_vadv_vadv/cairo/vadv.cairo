// outputs: mutated utens_stage
// output: tensor<3,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, HALF, ONE};
const QUARTER: Q32 = Q32 { bits: 1073741824 };

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (ni, nj, nk, stage): (u32, u32, u32, Array<Q32>) = args.array3();
    let (ui, uj, uk, u): (u32, u32, u32, Array<Q32>) = args.array3();
    let (wi, wj, wk, w): (u32, u32, u32, Array<Q32>) = args.array3();
    let (pi, pj, pk, pos): (u32, u32, u32, Array<Q32>) = args.array3();
    let (ti, tj, tk, tens): (u32, u32, u32, Array<Q32>) = args.array3();
    let dtr: Q32 = args.scalar();
    assert!(ui == ni && uj == nj && uk == nk, "u shape mismatch");
    assert!(pi == ni && pj == nj && pk == nk, "pos shape mismatch");
    assert!(ti == ni && tj == nj && tk == nk, "tens shape mismatch");
    assert!(wi == ni + 1 && wj == nj, "wcon shape mismatch");
    let mut result = array![];
    for i in 0..ni {
        for j in 0..nj {
            // Each column is independent, as in the Rust implementation.
            assert!(nk >= 2, "index out of bounds");
            let start = (i * nj + j) * nk;
            let u = u.span().slice(start, nk);
            let pos = pos.span().slice(start, nk);
            let tens = tens.span().slice(start, nk);
            let stage = stage.span().slice(start, nk);
            let lo = w.span().slice((i * nj + j) * wk, wk);
            let hi = w.span().slice(((i + 1) * nj + j) * wk, wk);
            let mut ccol = array![];
            let mut dcol = array![];
            let gcv = QUARTER * (*hi.at(1) + *lo.at(1));
            let cs = gcv * HALF;
            let c = gcv * HALF;
            let bcol = dtr - c;
            let correction = -cs * (*u.at(1) - *u.at(0));
            let d = dtr * *pos.at(0) + *tens.at(0) + *stage.at(0) + correction;
            let divided = ONE / bcol;
            ccol.append(c * divided);
            dcol.append(d * divided);
            for k in 1..nk - 1 {
                let gav = -QUARTER * (*hi.at(k) + *lo.at(k));
                let gcv = QUARTER * (*hi.at(k + 1) + *lo.at(k + 1));
                let as_ = gav * HALF;
                let cs = gcv * HALF;
                let acol = gav * HALF;
                let c = gcv * HALF;
                let bcol = dtr - acol - c;
                let correction = -as_ * (*u.at(k - 1) - *u.at(k))
                    - cs * (*u.at(k + 1) - *u.at(k));
                let d = dtr * *pos.at(k) + *tens.at(k) + *stage.at(k) + correction;
                let divided = ONE / (bcol - *ccol.at(k - 1) * acol);
                ccol.append(c * divided);
                dcol.append((d - *dcol.at(k - 1) * acol) * divided);
            };
            let k = nk - 1;
            let gav = -QUARTER * (*hi.at(k) + *lo.at(k));
            let as_ = gav * HALF;
            let acol = gav * HALF;
            let bcol = dtr - acol;
            let correction = -as_ * (*u.at(k - 1) - *u.at(k));
            let d = dtr * *pos.at(k) + *tens.at(k) + *stage.at(k) + correction;
            let divided = ONE / (bcol - *ccol.at(k - 1) * acol);
            let mut data_col = (d - *dcol.at(k - 1) * acol) * divided;
            let mut reverse = array![dtr * (data_col - *pos.at(k))];
            let mut k = k;
            while k > 0 {
                k -= 1;
                data_col = *dcol.at(k) - *ccol.at(k) * data_col;
                reverse.append(dtr * (data_col - *pos.at(k)));
            };
            for k in 0..nk { result.append(*reverse.at(nk - 1 - k)); };
        };
    };
    let mut out = args.output();
    tensor(ref out, array![ni, nj, nk].span(), result.span());
    out
}
