//! Pythran blacksholes: Black-Scholes call and put prices of the first
//! `nb_opt` options, with the polynomial approximation of the normal density
//! term.
//!
//! ```python
//! def BlackScholes(stock_price: Integer[Array, '99'], option_strike: Integer[Array, '99'], option_years: Integer[Array, '99'], Riskfree: Float, Volatility: Float, nb_opt: Integer):
//!     RSQRT2PI = 1 / math.sqrt(math.pi * 2)
//!     A1 = 0.31938153
//!     A2 = -0.356563782
//!     A3 = 1.781477937
//!     A4 = -1.821255978
//!     A5 = 1.330274429
//!     call_result = []
//!     put_result = []
//!     for opt in range(0, nb_opt):
//!         sqrtT = math.sqrt(option_years[opt])
//!         d1 = math.log(stock_price[opt] / option_strike[opt])
//!         d1 += (Riskfree + 0.5 * Volatility * Volatility) * option_years[opt]
//!         d1 /= Volatility * sqrtT
//!         d2 = d1 - Volatility * sqrtT
//!         K = 1.0 / (1.0 + 0.2316419 * abs(d1))
//!         CNDD1 = RSQRT2PI * math.exp(-0.5 * d1 * d1) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
//!         K = 1.0 / (1.0 + 0.2316419 * abs(d2))
//!         CNDD2 = RSQRT2PI * math.exp(-0.5 * d2 * d2) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
//!         expRT = math.exp(-Riskfree * option_years[opt])
//!         call_result.append(stock_price[opt] * CNDD1 - option_strike[opt] * expRT * CNDD2)
//!         put_result.append(option_strike[opt] * expRT * (1.0 - CNDD2) - stock_price[opt] * (1.0 - CNDD1))
//!     return (call_result, put_result)
//! ```
use crate::prelude::*;

/// `1 / math.sqrt(math.pi * 2)`, quantized.
const RSQRT2PI: Q32 = Q32::from_bits(1_713_444_047);
const A1: Q32 = Q32::from_bits(1_371_733_226); // 0.31938153
const A2: Q32 = Q32::from_bits(-1_531_429_783); // -0.356563782
const A3: Q32 = Q32::from_bits(7_651_389_478); // 1.781477937
const A4: Q32 = Q32::from_bits(-7_822_234_863); // -1.821255978
const A5: Q32 = Q32::from_bits(5_713_485_167); // 1.330274429
const HALF: Q32 = Q32::from_bits(1 << 31); // 0.5
const K_COEF: Q32 = Q32::from_bits(994_894_385); // 0.2316419

/// `RSQRT2PI * exp(-0.5 * d * d) * (K * (A1 + K * (... + K * A5)))` with
/// `K = 1 / (1 + 0.2316419 * |d|)`.
fn cnd(d: Q32) -> Q32 {
    let k = Q32::ONE / (Q32::ONE + K_COEF * d.abs());
    let poly = k * (A1 + k * (A2 + k * (A3 + k * (A4 + k * A5))));
    RSQRT2PI * (-HALF * d * d).exp() * poly
}

#[allow(non_snake_case)]
pub mod BlackScholes {
    use super::*;

    /// `(call_result, put_result)`, `nb_opt` prices each.
    pub type Output = (Vec<Q32>, Vec<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let stock_price: Array1<i64> = args.array1();
        let option_strike: Array1<i64> = args.array1();
        let option_years: Array1<i64> = args.array1();
        let riskfree: Q32 = args.scalar();
        let volatility: Q32 = args.scalar();
        let nb_opt: i64 = args.scalar();
        assert!(volatility != Q32::ZERO, "requires Volatility != 0.0");
        assert!((0..=1024).contains(&nb_opt), "requires 0 <= nb_opt <= 1024");

        let (mut call_result, mut put_result) = (Vec::new(), Vec::new());
        for opt in 0..nb_opt as usize {
            let (stock, strike, years) = (stock_price[opt], option_strike[opt], option_years[opt]);
            let vol_sqrt_t = volatility * Q32::from_int(years).sqrt();
            // `stock / strike` divides two integers: a float division.
            let mut d1 = (Q32::from_int(stock) / Q32::from_int(strike)).ln();
            d1 += (riskfree + HALF * volatility * volatility) * years;
            d1 /= vol_sqrt_t;
            let d2 = d1 - vol_sqrt_t;
            let (cndd1, cndd2) = (cnd(d1), cnd(d2));
            let exp_rt = (-riskfree * years).exp();
            call_result.push(cndd1 * stock - exp_rt * strike * cndd2);
            put_result.push(exp_rt * strike * (Q32::ONE - cndd2) - (Q32::ONE - cndd1) * stock);
        }
        (call_result, put_result)
    }
}
