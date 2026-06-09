from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import math

@zk_circuit
@requires(lambda stock_price, option_strike, option_years, Riskfree, Volatility, nb_opt: Volatility != 0.0)
@requires(lambda stock_price, option_strike, option_years, Riskfree, Volatility, nb_opt: 0 <= nb_opt <= 1024)
def BlackScholes(stock_price: Float[Array, 'N<=1024'], option_strike: Float[Array, 'N<=1024'], option_years: Float[Array, 'N<=1024'], Riskfree: Float, Volatility: Float, nb_opt: Integer):
    RSQRT2PI = 1 / math.sqrt(math.pi * 2)
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    call_result = []
    put_result = []
    for opt in range(0, nb_opt):
        sqrtT = math.sqrt(option_years[opt])
        d1 = math.log(stock_price[opt] / option_strike[opt])
        d1 += (Riskfree + 0.5 * Volatility * Volatility) * option_years[opt]
        d1 /= Volatility * sqrtT
        d2 = d1 - Volatility * sqrtT
        K = 1.0 / (1.0 + 0.2316419 * abs(d1))
        CNDD1 = RSQRT2PI * math.exp(-0.5 * d1 * d1) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
        K = 1.0 / (1.0 + 0.2316419 * abs(d2))
        CNDD2 = RSQRT2PI * math.exp(-0.5 * d2 * d2) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
        expRT = math.exp(-Riskfree * option_years[opt])
        call_result.append(stock_price[opt] * CNDD1 - option_strike[opt] * expRT * CNDD2)
        put_result.append(option_strike[opt] * expRT * (1.0 - CNDD2) - stock_price[opt] * (1.0 - CNDD1))
    return (call_result, put_result)
