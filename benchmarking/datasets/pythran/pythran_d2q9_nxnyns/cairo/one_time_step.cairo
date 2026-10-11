//! D2Q9 streaming and moment relaxation, including the reference's transposed store.
// output: tensor<3,q32>
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait};
use crate::np::np_index;
const C0: Q32 = Q32 { bits: 715827883 };
const C1: Q32 = Q32 { bits: 477218588 };
const C2: Q32 = Q32 { bits: 238609294 };
const C3: Q32 = Q32 { bits: 119304647 };
const C4: Q32 = Q32 { bits: 357913941 };
const QUARTER: Q32 = Q32 { bits: 1073741824 };
const THREE_HALVES: Q32 = Q32 { bits: 6442450944 };
const OMEGA_E: Q32 = Q32 { bits: 4858560290 };
const OMEGA_XY: Q32 = Q32 { bits: 7977279525 };

fn m2f_loc(m: Span<Q32>) -> Array<Q32> {
    array![
        C1 * *m.at(0) - C1 * *m.at(3) + C1 * *m.at(4),
        C1 * *m.at(0) + C0 * *m.at(1) - C3 * *m.at(3) - C2 * *m.at(4) - C0 * *m.at(5) + QUARTER * *m.at(7),
        C1 * *m.at(0) + C0 * *m.at(2) - C3 * *m.at(3) - C2 * *m.at(4) - C0 * *m.at(6) - QUARTER * *m.at(7),
        C1 * *m.at(0) - C0 * *m.at(1) - C3 * *m.at(3) - C2 * *m.at(4) + C0 * *m.at(5) + QUARTER * *m.at(7),
        C1 * *m.at(0) - C0 * *m.at(2) - C3 * *m.at(3) - C2 * *m.at(4) + C0 * *m.at(6) - QUARTER * *m.at(7),
        C1 * *m.at(0)
            + C0 * *m.at(1)
            + C0 * *m.at(2)
            + C2 * *m.at(3)
            + C3 * *m.at(4)
            + C4 * *m.at(5)
            + C4 * *m.at(6)
            + QUARTER * *m.at(8),
        C1 * *m.at(0) - C0 * *m.at(1) + C0 * *m.at(2) + C2 * *m.at(3) + C3 * *m.at(4) - C4 * *m.at(5) + C4 * *m.at(6)
            - QUARTER * *m.at(8),
        C1 * *m.at(0) - C0 * *m.at(1) - C0 * *m.at(2) + C2 * *m.at(3) + C3 * *m.at(4) - C4 * *m.at(5) - C4 * *m.at(6)
            + QUARTER * *m.at(8),
        C1 * *m.at(0) + C0 * *m.at(1) - C0 * *m.at(2) + C2 * *m.at(3) + C3 * *m.at(4) + C4 * *m.at(5)
            - C4 * *m.at(6)
            - QUARTER * *m.at(8),
    ]
}

fn f2m_loc(f: Span<Q32>) -> Array<Q32> {
    array![
        *f.at(0) + *f.at(1) + *f.at(2) + *f.at(3) + *f.at(4) + *f.at(5) + *f.at(6) + *f.at(7) + *f.at(8),
        *f.at(1) - *f.at(3) + *f.at(5) - *f.at(6) - *f.at(7) + *f.at(8),
        *f.at(2) - *f.at(4) + *f.at(5) + *f.at(6) - *f.at(7) - *f.at(8),
        (*f.at(0)).mul_int(-4) - *f.at(1) - *f.at(2) - *f.at(3) - *f.at(4) + (*f.at(5)).mul_int(2) + (*f.at(6)).mul_int(2) + (*f.at(7)).mul_int(2) + (*f.at(8)).mul_int(2),
        (*f.at(0)).mul_int(4) - (*f.at(1)).mul_int(2) - (*f.at(2)).mul_int(2) - (*f.at(3)).mul_int(2) - (*f.at(4)).mul_int(2) + *f.at(5) + *f.at(6) + *f.at(7) + *f.at(8),
        (*f.at(1)).mul_int(-2) + (*f.at(3)).mul_int(2) + *f.at(5) - *f.at(6) - *f.at(7) + *f.at(8),
        (*f.at(2)).mul_int(-2) + (*f.at(4)).mul_int(2) + *f.at(5) + *f.at(6) - *f.at(7) - *f.at(8),
        *f.at(1) - *f.at(2) + *f.at(3) - *f.at(4),
        *f.at(5) - *f.at(6) + *f.at(7) - *f.at(8),
    ]
}

fn relaxation_loc(m: Span<Q32>) -> Array<Q32> {
    array![*m.at(0), *m.at(1), *m.at(2),
        *m.at(3) + OMEGA_E * ((*m.at(0)).mul_int(-2) + (*m.at(1)).mul_int(3) * *m.at(1) + (*m.at(2)).mul_int(3) * *m.at(2) - *m.at(3)),
        *m.at(4) + OMEGA_E * (*m.at(0) + THREE_HALVES * *m.at(1) * *m.at(1) + THREE_HALVES * *m.at(2) * *m.at(2) - *m.at(4)),
        *m.at(5) + OMEGA_E * (-*m.at(1) - *m.at(5)),
        *m.at(6) + OMEGA_E * (-*m.at(2) - *m.at(6)),
        *m.at(7) + OMEGA_XY * (*m.at(1) * *m.at(1) - *m.at(2) * *m.at(2) - *m.at(7)),
        *m.at(8) + OMEGA_XY * (*m.at(1) * *m.at(2) - *m.at(8)),
    ]
}

fn get(ref data: Felt252Dict<felt252>, shape: (u32, u32, u32), i: u32, j: u32, k: u32) -> Q32 {
    let (nx, ny, ns) = shape;
    assert!(i < nx && j < ny && k < ns, "population index out of bounds");
    Q32 { bits: data.get(((i * ny + j) * ns + k).into()).try_into().unwrap() }
}
fn put(ref data: Felt252Dict<felt252>, shape: (u32, u32, u32), i: u32, j: u32, k: u32, value: Q32) {
    let (nx, ny, ns) = shape;
    assert!(i < nx && j < ny && k < ns, "population index out of bounds");
    data.insert(((i * ny + j) * ns + k).into(), value.bits.into());
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (nx, ny, ns, f1): (u32, u32, u32, Array<Q32>) = args.array3();
    let (ox, oy, os, f2): (u32, u32, u32, Array<Q32>) = args.array3();
    let shape = (nx, ny, ns);
    let output_shape = (ox, oy, os);
    let mut src: Felt252Dict<felt252> = Default::default();
    let mut dst: Felt252Dict<felt252> = Default::default();
    for k in 0..f1.len() { src.insert(k.into(), (*f1.at(k)).bits.into()); };
    for k in 0..f2.len() { dst.insert(k.into(), (*f2.at(k)).bits.into()); };
    // Preserve sequential boundary writes, including axes of length two.
    for j in 0..ny {
        for k in 0..ns {
            let x: i64 = nx.into();
            let value = get(ref src, shape, np_index(x - 2, nx), j, k);
            put(ref src, shape, 0, j, k, value);
            let value = get(ref src, shape, 1, j, k);
            put(ref src, shape, np_index(x - 1, nx), j, k, value);
        };
    };
    for i in 0..nx {
        for k in 0..ns {
            let y: i64 = ny.into();
            let value = get(ref src, shape, i, np_index(y - 2, ny), k);
            put(ref src, shape, i, 0, k, value);
            let value = get(ref src, shape, i, 1, k);
            put(ref src, shape, i, np_index(y - 1, ny), k, value);
        };
    };
    if nx > 2 && ny > 2 {
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let f = array![
                    get(ref src, shape, i, j, 0),
                    get(ref src, shape, i - 1, j, 1),
                    get(ref src, shape, i, j - 1, 2),
                    get(ref src, shape, i + 1, j, 3),
                    get(ref src, shape, i, j + 1, 4),
                    get(ref src, shape, i - 1, j - 1, 5),
                    get(ref src, shape, i + 1, j - 1, 6),
                    get(ref src, shape, i + 1, j + 1, 7),
                    get(ref src, shape, i - 1, j + 1, 8),
                ];
                let m = f2m_loc(f.span());
                let relaxed = relaxation_loc(m.span());
                let populations = m2f_loc(relaxed.span());
                for k in 0..9_u32 { put(ref dst, output_shape, j, i, k, *populations.at(k)); };
            };
        };
    };
    let mut result: Array<Q32> = array![];
    for k in 0..f2.len() { result.append(Q32 { bits: dst.get(k.into()).try_into().unwrap() }); };
    let mut out = args.output();
    tensor(ref out, array![ox, oy, os].span(), result.span());
    out
}
