import numpy as np
from zinnia import NDArray

# ---------------------------
# Fixed-point configuration
# ---------------------------

np.random.seed(42)


FXP_FRAC_BITS = 64
FXP_ONE = 1 << FXP_FRAC_BITS
FXP_HALF = 1 << (FXP_FRAC_BITS - 1)

def to_fxp(arr: np.ndarray) -> np.ndarray:
    """Quantize float64 -> fixed-point integers with scale 2^64 (round-to-nearest)."""
    return np.rint(arr.astype(float) * FXP_ONE).astype(object)

def from_fxp(arr_q: np.ndarray) -> np.ndarray:
    """Dequantize fixed-point integers -> float64."""
    return np.asarray(arr_q, dtype=float) / FXP_ONE

def fxp_mul(a: int, b: int) -> int:
    """(a * b) / 2^64 with round-to-nearest, sign-aware."""
    prod = a * b
    return (prod + FXP_HALF) // FXP_ONE if prod >= 0 else (prod - FXP_HALF) // FXP_ONE

def fxp_sq(a: int) -> int:
    """a^2 / 2^64 (round-to-nearest). Non-negative result."""
    prod = a * a
    return (prod + FXP_HALF) // FXP_ONE

def fxp_div(a: int, b: int) -> int:
    """a / b in fixed-point (both fixed-point), returns fixed-point (round-to-nearest)."""
    if b == 0:
        raise ZeroDivisionError("fxp_div: division by zero")
    # Compute ((a << 64) / b) with rounding toward nearest
    num = a << FXP_FRAC_BITS
    if (num ^ b) >= 0:  # same sign
        return (num + (abs(b) // 2)) // b
    else:
        return (num - (abs(b) // 2)) // b

def fxp_sqrt(x: int, iters: int = 24) -> int:
    """
    Fixed-point sqrt: returns y ~ sqrt(x) in fixed-point, using Newton-Raphson.
    x, y are fixed-point values (scaled by 2^64).
    """
    if x <= 0:
        return 0
    # Initial guess: 1.0 in fixed-point, scaled roughly by magnitude of x.
    # Use bit-length heuristic to pick y0 ≈ 2^{(bl-64)/2} in fixed-point.
    bl = x.bit_length()
    # Target sqrt scale: (bl - FXP_FRAC_BITS)/2 in integer bits
    shift = max(0, (bl - FXP_FRAC_BITS) // 2)
    y = FXP_ONE << shift  # fixed-point 2^shift
    # Newton iterations: y_{k+1} = 0.5 * (y + x / y)
    half = FXP_ONE >> 1
    for _ in range(iters):
        try:
            div = fxp_div(x, y)
        except ZeroDivisionError:
            div = x  # fallback, shouldn't happen with y>0
        y = fxp_mul(half, (y + div))
    return y

def normalize_rows_float(X: np.ndarray) -> np.ndarray:
    norms = np.sqrt((X * X).sum(axis=-1)).reshape((X.shape[0], 1))
    return X / norms

def normalize_rows_fixed_point(X: np.ndarray) -> np.ndarray:
    """
    Integer-only row-wise normalization in fixed-point:
    y_i = x_i / ||x||_2 where all ops are in 2^-64 fixed-point.
    """
    Xq = to_fxp(np.asarray(X, dtype=float))
    n, d = Xq.shape
    Yq = np.zeros_like(Xq, dtype=object)

    for i in range(n):
        # ||x||^2 = sum_j x_j^2  (fixed-point)
        s = 0
        for j in range(d):
            s += fxp_sq(int(Xq[i, j]))
        # ||x|| = sqrt(s)  (fixed-point)
        norm_q = fxp_sqrt(int(s))
        if norm_q == 0:
            # preserve zeros if the row is all zeros
            for j in range(d):
                Yq[i, j] = 0
        else:
            for j in range(d):
                Yq[i, j] = fxp_div(int(Xq[i, j]), int(norm_q))

    return from_fxp(Yq)

def verify_solution(X: NDArray[float, 5, 4], result: NDArray[float, 5, 4]):
    # Float64 baseline
    float_result = normalize_rows_float(X)

    # Fixed-point path
    fxp_result = normalize_rows_fixed_point(X)

    # Error reporting
    diff = fxp_result - float_result
    abs_err = np.abs(diff)
    rel_err = abs_err / (np.abs(float_result) + 1e-18)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    print("Abs error (per entry):\n", abs_err)
    print("Rel error (per entry):\n", rel_err)
    print("RMSE:", rmse)

    # Keep the original intended check for the float path
    assert np.allclose(result, float_result, atol=1e-12, rtol=1e-12)

# ---------------------------
# Randomized test harness
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # Generate non-zero rows to avoid div-by-zero in normalization
    X = rng.uniform(-5, 5, size=(5, 4))
    # Ensure no zero-norm row (very unlikely already, but make robust)
    for i in range(5):
        if np.linalg.norm(X[i]) < 1e-12:
            X[i, 0] = 1.0

    # Expected result from float path
    expected = X / np.sqrt((X * X).sum(axis=-1)).reshape((5, 1))

    print("=== Row-wise L2 normalization: float vs fixed-point (2^-64) ===")
    verify_solution(X, expected)

# 3.1646225226420967e-17