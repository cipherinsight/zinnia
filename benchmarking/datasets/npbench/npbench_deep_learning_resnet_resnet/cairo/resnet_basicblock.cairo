//! ResNet basic block: valid NHWC convolutions, batch-axis normalization and residual ReLU.
// output: tensor<4,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};

// NHWC values and K x K x Cin x Cout weights share this local view.
#[derive(Copy, Drop)]
struct T4 {
    n: u32,
    h: u32,
    w: u32,
    c: u32,
    data: Span<Q32>,
}
fn at(t: T4, b: u32, i: u32, j: u32, c: u32) -> Q32 {
    // All callers establish axis bounds through their loops and shape checks.
    *t.data.at(((b * t.h + i) * t.w + j) * t.c + c)
}
fn bi(dim: u32, i: u32) -> u32 {
    if dim == 1 {
        0
    } else {
        i
    }
}
fn broadcast(a: u32, b: u32) -> u32 {
    assert!(a == b || a == 1 || b == 1, "residual shape mismatch");
    if a == 1 {
        b
    } else {
        a
    }
}
fn conv(input: T4, weights: T4) -> T4 {
    let k = weights.n;
    assert!(weights.h == k && weights.w == input.c, "conv weights mismatch");
    assert!(k <= input.h + 1 && k <= input.w + 1, "negative dimensions");
    let h = input.h + 1 - k;
    let w = input.w + 1 - k;
    let mut data = array![];
    for b in 0..input.n {
        for i in 0..h {
            for j in 0..w {
                for co in 0..weights.c {
                    let mut sum = ZERO;
                    // np.sum of a product array: round each product separately.
                    for di in 0..k {
                        for dj in 0..k {
                            for ci in 0..input.c {
                                sum = sum
                                    + at(input, b, i + di, j + dj, ci)
                                        * at(weights, di, dj, ci, co);
                            };
                        };
                    }
                    data.append(sum);
                };
            };
        };
    }
    T4 { n: input.n, h, w, c: weights.c, data: data.span() }
}
fn norm(x: T4, relu: bool) -> T4 {
    let lanes = x.h * x.w * x.c;
    let mut means = array![];
    let mut denoms = array![];
    for i in 0..lanes {
        let mut sum = ZERO;
        for b in 0..x.n {
            sum = sum + *x.data.at(b * lanes + i);
        }
        let mean = sum.div_int(x.n.into());
        let mut variance = ZERO;
        for b in 0..x.n {
            let d = *x.data.at(b * lanes + i) - mean;
            variance = variance + d * d;
        }
        // The Rust reference uses sqrt(std + eps), not sqrt(variance + eps).
        let denominator = (variance.div_int(x.n.into()).sqrt() + Q32 { bits: 42950 }).sqrt();
        means.append(mean);
        denoms.append(denominator);
    }
    let mut data = array![];
    for b in 0..x.n {
        for i in 0..lanes {
            let v = (*x.data.at(b * lanes + i) - *means.at(i)) / *denoms.at(i);
            data.append(if relu && v < ZERO {
                ZERO
            } else {
                v
            });
        };
    }
    T4 { data: data.span(), ..x }
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (n, h, w, c, input): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let (k1, kw1, ci1, co1, v1): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let (k2, kw2, ci2, co2, v2): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let (k3, kw3, ci3, co3, v3): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let input = T4 { n, h, w, c, data: input.span() };
    let first = conv(input, T4 { n: k1, h: kw1, w: ci1, c: co1, data: v1.span() });
    assert!(
        (first.h == h || first.h == 1) && (first.w == w || first.w == 1),
        "padding assignment shape mismatch",
    );
    let mut padded = array![];
    for b in 0..n {
        for i in 0..h + 2 {
            for j in 0..w + 2 {
                for co in 0..co1 {
                    let v = if i > 0 && i <= h && j > 0 && j <= w {
                        at(
                            first,
                            b,
                            if first.h == 1 {
                                0
                            } else {
                                i - 1
                            },
                            if first.w == 1 {
                                0
                            } else {
                                j - 1
                            },
                            co,
                        )
                    } else {
                        ZERO
                    };
                    padded.append(v);
                };
            };
        };
    }
    let x = norm(T4 { n, h: h + 2, w: w + 2, c: co1, data: padded.span() }, true);
    let x = norm(conv(x, T4 { n: k2, h: kw2, w: ci2, c: co2, data: v2.span() }), true);
    let x = norm(conv(x, T4 { n: k3, h: kw3, w: ci3, c: co3, data: v3.span() }), false);
    let rn = broadcast(x.n, n);
    let rh = broadcast(x.h, h);
    let rw = broadcast(x.w, w);
    let rc = broadcast(x.c, c);
    let mut result = array![];
    for b in 0..rn {
        for i in 0..rh {
            for j in 0..rw {
                for co in 0..rc {
                    let v = at(x, bi(x.n, b), bi(x.h, i), bi(x.w, j), bi(x.c, co))
                        + at(input, bi(n, b), bi(h, i), bi(w, j), bi(c, co));
                    result.append(if v > ZERO {
                        v
                    } else {
                        ZERO
                    });
                };
            };
        };
    }
    let mut out = args.output();
    tensor(ref out, array![rn, rh, rw, rc].span(), result.span());
    out
}
