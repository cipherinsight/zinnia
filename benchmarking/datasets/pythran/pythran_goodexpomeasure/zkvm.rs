//! Pythran goodexpomeasure: the "good exposure" quality measure of an RGB
//! image, `exp(-(c - 128)^2 / sigma)` multiplied over the three channels and
//! quantized to `uint8`.
//!
//! ```python
//! @requires(lambda inRGB, sigma: sigma != 0)
//! def goodExpoMeasure(inRGB: Integer[Array, '20 20 20'], sigma: Float):
//!     R = inRGB[0, :, :].astype(numpy.float64)
//!     G = inRGB[1, :, :].astype(numpy.float64)
//!     B = inRGB[2, :, :].astype(numpy.float64)
//!     goodExpoR = numpy.exp(-(R - 128) ** 2 / sigma)
//!     goodExpoG = numpy.exp(-(G - 128) ** 2 / sigma)
//!     goodExpoB = numpy.exp(-(B - 128) ** 2 / sigma)
//!     goodExpo = goodExpoR * goodExpoG * goodExpoB
//!     goodExpo = (numpy.round(goodExpo, 2) * (2 ** 8 - 1)).astype(numpy.uint8)
//!     return goodExpo
//! ```
use crate::prelude::*;

/// `np.rint`: the nearest integer, ties to even.
fn rint(x: Q32) -> i64 {
    let bits = x.to_bits();
    let (floor, fraction) = (bits >> 32, bits & 0xFFFF_FFFF);
    let half = 1 << 31;
    if fraction > half || (fraction == half && floor & 1 == 1) {
        floor + 1
    } else {
        floor
    }
}

#[allow(non_snake_case)]
pub mod goodExpoMeasure {
    use super::*;

    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let in_rgb: Array3<i64> = args.array3();
        let sigma: Q32 = args.scalar();
        assert!(sigma != Q32::ZERO, "requires sigma != 0");
        // `-(c - 128) ** 2 / sigma` is `(-((c - 128) ** 2)) / sigma`.
        let good_expo = |channel: usize| {
            in_rgb.index_axis(Axis(0), channel).mapv(|v| {
                let d = Q32::from_int(v) - Q32::from_int(128);
                (-(d * d) / sigma).exp()
            })
        };
        let (r, g, b) = (good_expo(0), good_expo(1), good_expo(2));
        Zip::from(&r).and(&g).and(&b).map_collect(|&r, &g, &b| {
            // numpy.round(x, 2) is rint(100 x) / 100. The truncated product
            // (k / 100) * 255 is computed exactly as 255 k / 100: NumPy's float
            // product lands on that integer for every k in 0..=100, while the
            // Q32 quotient k / 100 would fall just below it at k = 20, 40, 60, 80.
            let k = rint(r * g * b * 100);
            k * 255 / 100
        })
    }
}
