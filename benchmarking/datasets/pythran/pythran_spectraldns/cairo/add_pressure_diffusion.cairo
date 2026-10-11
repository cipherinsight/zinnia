//! Returned pressure projection; the Rust reference omits the dead dU update.
// unused parameters: U_hat, K2, K, nu
// output: tensor<3,c32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, C32, C32Trait};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (dn, dx, dy, dz, du): (u32, u32, u32, u32, Array<C32>) = args.array4();
    let (px, py, pz, _p): (u32, u32, u32, Array<C32>) = args.array3();
    let (kn, kx, ky, kz, k): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    assert!(dx == px && dy == py && dz == pz, "could not broadcast into P_hat");
    let size = px * py * pz;
    if size > 0 {
        assert!(dn >= 3 && kn >= 3 && kx >= px && ky >= py && kz >= pz, "pressure index out of bounds");
    };
    let mut result = array![];
    for x in 0..px {
        for y in 0..py {
            for z in 0..pz {
                let p = (x * py + y) * pz + z;
                let q = (x * ky + y) * kz + z;
                let ks = kx * ky * kz;
                let t0 = (*du.at(p)).scale(*k.at(q));
                let t1 = (*du.at(size + p)).scale(*k.at(ks + q));
                let t2 = (*du.at(2 * size + p)).scale(*k.at(2 * ks + q));
                result.append(t0 + t1 + t2);
            };
        };
    };
    let mut out = args.output();
    tensor(ref out, array![px, py, pz].span(), result.span());
    out
}
