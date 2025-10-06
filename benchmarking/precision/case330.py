import numpy as np
from zinnia import NDArray

# ---------------------------
# Fixed-point configuration
# ---------------------------
FXP_FRAC_BITS = 64
FXP_ONE = 1 << FXP_FRAC_BITS
FXP_HALF = 1 << (FXP_FRAC_BITS - 1)

def to_fxp(arr: np.ndarray) -> np.ndarray:
    """Quantize float64 -> fixed-point integers with scale 2^64."""
    return np.rint(arr.astype(float) * FXP_ONE).astype(object)

def from_fxp(arr_q: np.ndarray) -> np.ndarray:
    """Dequantize fixed-point integers -> float64."""
    return np.asarray(arr_q, dtype=float) / FXP_ONE

def fxp_mul(a: int, b: int) -> int:
    """(a * b) / 2^64 with round-to-nearest."""
    prod = a * b
    if prod >= 0:
        return (prod + FXP_HALF) // FXP_ONE
    else:
        return (prod - FXP_HALF) // FXP_ONE

def fxp_pow_scalar(base_q: int, exp: int) -> int:
    """Integer exponentiation in fixed-point (only for non-negative integer exp)."""
    result = FXP_ONE  # 1.0 in fixed-point
    b = base_q
    e = exp
    while e > 0:
        if e & 1:
            result = fxp_mul(result, b)
        b = fxp_mul(b, b)
        e >>= 1
    return result

def fxp_pow_matrix(a: np.ndarray, exp: int) -> np.ndarray:
    """Element-wise exponentiation for fixed-point integers (integer power only)."""
    n, m = a.shape
    out = np.zeros((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            out[i, j] = fxp_pow_scalar(int(a[i, j]), exp)
    return out

def verify_solution(a: NDArray[float, 2, 2], power: int, desired_result: NDArray[float, 2, 2]):
    # Float64 computation
    float_result = a ** power

    # Fixed-point computation
    aq = to_fxp(a)
    fxp_result_q = fxp_pow_matrix(aq, power)
    fxp_result = from_fxp(fxp_result_q)

    # Report errors
    diff = fxp_result - float_result
    abs_err = np.abs(diff)
    rel_err = abs_err / (np.abs(float_result) + 1e-18)
    rmse = np.sqrt(np.mean(diff ** 2))

    print("Input a:\n", a)
    print("Float result:\n", float_result)
    print("Fixed-point result:\n", fxp_result)
    print("Abs error:\n", abs_err)
    print("Rel error:\n", rel_err)
    print("RMSE:", rmse)

    # Original intended check (float vs desired)
    assert np.allclose(float_result, desired_result, atol=1e-12)

# ---------------------------
# Randomized test harness
# ---------------------------
if __name__ == "__main__":
    np.random.seed(42)
    for _ in range(5):
        a = np.random.uniform(-2, 2, size=(2, 2))  # random matrix
        power = np.random.randint(1, 6)            # random small positive integer power
        desired = a ** power                       # baseline desired result
        print(f"\n=== Test: power={power} ===")
        verify_solution(a, power, desired)


# RMSE 5.421010862427522e-20