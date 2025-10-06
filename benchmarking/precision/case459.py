import numpy as np
from zinnia import NDArray

# ---------------------------
# Fixed-point configuration
# ---------------------------
FXP_FRAC_BITS = 64
FXP_ONE = 1 << FXP_FRAC_BITS
FXP_HALF = 1 << (FXP_FRAC_BITS - 1)

def to_fxp(arr: np.ndarray) -> np.ndarray:
    """Quantize float64 -> fixed-point integers with scale 2^64 (round-to-nearest)."""
    return np.rint(arr.astype(float) * FXP_ONE).astype(object)

def from_fxp(arr_q: np.ndarray) -> np.ndarray:
    """Dequantize fixed-point integers -> float64."""
    return np.asarray(arr_q, dtype=float) / FXP_ONE

def fxp_add(a: int, b: int) -> int:
    return a + b

def fxp_div_by_int(a: int, k: int) -> int:
    """Divide fixed-point value `a` by positive integer k (round-to-nearest)."""
    if a >= 0:
        return (a + k // 2) // k
    else:
        return (a - k // 2) // k

def average_fixed_point(A: np.ndarray) -> float:
    """Compute average of array A in fixed-point, return as float."""
    Aq = to_fxp(np.asarray(A, dtype=float))
    s = 0
    for v in Aq:
        s = fxp_add(s, int(v))
    avg_q = fxp_div_by_int(s, len(A))
    return float(avg_q) / FXP_ONE

def verify_solution(A: NDArray[float, 4], AVG: float):
    # Float64 baseline
    avg_f = np.sum(np.asarray(A).astype(float), axis=0) / len(A)

    # Fixed-point computation
    avg_q = average_fixed_point(A)

    # Report error
    abs_err = abs(avg_q - avg_f)
    rel_err = abs_err / (abs(avg_f) + 1e-18)
    print(f"A: {A}")
    print(f"Float64 average: {avg_f}")
    print(f"Fixed-point average: {avg_q}")
    print(f"Abs error: {abs_err}")
    print(f"Rel error: {rel_err}")

    # Original intended check for float path
    assert np.allclose(AVG, avg_f, atol=1e-12, rtol=1e-12)

# ---------------------------
# Randomized test harness
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for _ in range(5):
        A = rng.uniform(-10, 10, size=(4,))
        AVG = np.sum(A.astype(float)) / len(A)
        print("\n=== New Test ===")
        verify_solution(A, AVG)

# Relative error 1.7478424766349322e-16