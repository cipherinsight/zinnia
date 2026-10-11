//! Shape-generic Q32 port of zkvm.rs::kernel.
// outputs: mutated A, w, x
// output: (tensor<2,q32>,tensor<1,q32>,tensor<1,q32>)
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let alpha: Q32 = args.scalar();
    let beta: Q32 = args.scalar();
    let (r, c, a): (u32, u32, Array<Q32>) = args.array2();
    let u1: Array<Q32> = args.array1();
    let v1: Array<Q32> = args.array1();
    let u2: Array<Q32> = args.array1();
    let v2: Array<Q32> = args.array1();
    let w: Array<Q32> = args.array1();
    let x: Array<Q32> = args.array1();
    let y: Array<Q32> = args.array1();
    let z: Array<Q32> = args.array1();
    assert!(
        u1.len() == r && u2.len() == r && v1.len() == c && v2.len() == c, "outer shape mismatch",
    );
    let mut updated_a = array![];
    for i in 0..r {
        for j in 0..c {
            updated_a.append(*a.at(i * c + j) + (*u1.at(i) * *v1.at(j) + *u2.at(i) * *v2.at(j)));
        };
    }
    let mut by = array![];
    for v in y {
        by.append(beta * v);
    }
    let ya = crate::np::vecmat(by.span(), updated_a.span(), r, c);
    assert!(x.len() == c && z.len() == c, "x shape mismatch");
    let mut updated_x = array![];
    for j in 0..c {
        updated_x.append(*x.at(j) + (*ya.at(j) + *z.at(j)));
    }
    let mut scaled_a = array![];
    for v in updated_a.span() {
        scaled_a.append(alpha * *v);
    }
    let ax = crate::np::matvec(scaled_a.span(), updated_x.span(), r, c);
    // ndarray += broadcasts the right-hand vector to w.
    assert!(r == w.len() || r == 1, "w shape mismatch");
    let mut updated_w = array![];
    for j in 0..w.len() {
        updated_w.append(*w.at(j) + *ax.at(if r == 1 {
            0
        } else {
            j
        }));
    }
    let mut out = args.output();
    tensor(ref out, array![r, c].span(), updated_a.span());
    tensor(ref out, array![w.len()].span(), updated_w.span());
    tensor(ref out, array![c].span(), updated_x.span());
    out
}
