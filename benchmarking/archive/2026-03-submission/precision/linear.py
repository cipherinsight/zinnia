import json
from typing import Tuple

from zinnia import NDArray, np

# ---------------------------
# Fixed-point configuration
# ---------------------------
FXP_FRAC_BITS = 64
FXP_ONE = 1 << FXP_FRAC_BITS
FXP_HALF = 1 << (FXP_FRAC_BITS - 1)

def to_fxp(arr: np.ndarray) -> np.ndarray:
    """Quantize float64 -> signed integer fixed point with scale 2^64 (round-to-nearest)."""
    return np.rint(arr.astype(float) * FXP_ONE).astype(object)

def from_fxp(arr_q: np.ndarray) -> np.ndarray:
    """Dequantize fixed-point -> float64."""
    return np.asarray(arr_q, dtype=float) / FXP_ONE

def fxp_add(a: int, b: int) -> int:
    return a + b

def fxp_sub(a: int, b: int) -> int:
    return a - b

def fxp_mul(a: int, b: int) -> int:
    """(a * b) / 2^64 with round-to-nearest, sign-aware."""
    prod = a * b
    if prod >= 0:
        return (prod + FXP_HALF) // FXP_ONE
    else:
        return (prod - FXP_HALF) // FXP_ONE

def fxp_sq(a: int) -> int:
    prod = a * a
    return (prod + FXP_HALF) // FXP_ONE

def fxp_div_by_int(a: int, k: int) -> int:
    """Divide fixed-point value `a` by positive integer k (round-to-nearest)."""
    if a >= 0:
        return (a + k // 2) // k
    else:
        return (a - k // 2) // k

def fxp_dot(vec_a_q: np.ndarray, vec_b_q: np.ndarray) -> int:
    """Dot product in fixed-point (returns fixed-point)."""
    s = 0
    for ai, bi in zip(vec_a_q, vec_b_q):
        s = fxp_add(s, fxp_mul(int(ai), int(bi)))
    return s

def kmeans_safe_arg():  # placeholder to avoid accidental unused warnings in some setups
    return None

def train_eval_float(
    training_x: NDArray[float, 10, 2],
    training_y: NDArray[float, 10],
    testing_x: NDArray[float, 2, 2],
    testing_y: NDArray[float, 2],
    lr: float = 0.02,
    iters: int = 100,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    n, d = training_x.shape
    w = np.zeros((d,), dtype=float)
    b = 0.0
    m = float(len(training_y))

    for _ in range(iters):
        preds = training_x @ w + b
        errs = preds - training_y
        dw = (1.0 / m) * (training_x.T @ errs)
        db = (1.0 / m) * np.sum(errs)
        w -= lr * dw
        b -= lr * db

    test_preds = testing_x @ w + b
    test_err = np.mean((test_preds - testing_y) ** 2)
    return w, b, test_preds, test_err

def train_eval_fixed_point(
    training_x: NDArray[float, 10, 2],
    training_y: NDArray[float, 10],
    testing_x: NDArray[float, 2, 2],
    testing_y: NDArray[float, 2],
    lr: float = 0.02,
    iters: int = 100,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    n, d = training_x.shape
    m_int = int(len(training_y))
    assert m_int > 0

    # Quantize data and hyperparams
    Xq = to_fxp(np.asarray(training_x, dtype=float))
    yq = to_fxp(np.asarray(training_y, dtype=float))
    Xtest_q = to_fxp(np.asarray(testing_x, dtype=float))
    ytest_q = to_fxp(np.asarray(testing_y, dtype=float))
    lr_q = int(np.rint(lr * FXP_ONE))

    wq = np.zeros((d,), dtype=object)
    bq = 0

    # GD loop
    for _ in range(iters):
        # preds = X @ w + b
        preds_q = np.zeros((n,), dtype=object)
        for i in range(n):
            preds_q[i] = fxp_add(fxp_dot(Xq[i], wq), bq)

        # errs = preds - y
        errs_q = np.zeros((n,), dtype=object)
        for i in range(n):
            errs_q[i] = fxp_sub(int(preds_q[i]), int(yq[i]))

        # dw = (1/m) * (X^T @ errs)
        dw_q = np.zeros((d,), dtype=object)
        for j in range(d):
            s = 0
            for i in range(n):
                s = fxp_add(s, fxp_mul(int(Xq[i][j]), int(errs_q[i])))
            dw_q[j] = fxp_div_by_int(s, m_int)

        # db = (1/m) * sum(errs)
        s_err = 0
        for i in range(n):
            s_err = fxp_add(s_err, int(errs_q[i]))
        db_q = fxp_div_by_int(s_err, m_int)

        # w -= lr * dw ; b -= lr * db
        for j in range(d):
            wq[j] = fxp_sub(int(wq[j]), fxp_mul(lr_q, int(dw_q[j])))
        bq = fxp_sub(int(bq), fxp_mul(lr_q, int(db_q)))

    # Test predictions
    test_preds_q = np.zeros((len(testing_y),), dtype=object)
    for i in range(len(testing_y)):
        test_preds_q[i] = fxp_add(fxp_dot(Xtest_q[i], wq), bq)

    # MSE in fixed-point, then dequantize
    mse_q_sum = 0
    for i in range(len(testing_y)):
        e_q = fxp_sub(int(test_preds_q[i]), int(ytest_q[i]))
        mse_q_sum = fxp_add(mse_q_sum, fxp_sq(e_q))
    mse_q = fxp_div_by_int(mse_q_sum, int(len(testing_y)))

    # Dequantize parameters and outputs for reporting
    w = from_fxp(wq)
    b = float(bq) / FXP_ONE
    test_preds = from_fxp(test_preds_q)
    test_err = float(mse_q) / FXP_ONE  # since errors were already squared in fxp

    return w, b, test_preds, test_err

def verify_solution(
        training_x: NDArray[float, 10, 2],
        training_y: NDArray[float, 10],
        testing_x: NDArray[float, 2, 2],
        testing_y: NDArray[float, 2]
):
    # Hyperparams mirror original
    LR = 0.02
    ITERS = 100

    # Float64 baseline
    w_f, b_f, preds_f, mse_f = train_eval_float(training_x, training_y, testing_x, testing_y, lr=LR, iters=ITERS)

    # Fixed-point (2^-64)
    w_q, b_q, preds_q, mse_q = train_eval_fixed_point(training_x, training_y, testing_x, testing_y, lr=LR, iters=ITERS)

    # Report
    print("Float64 MSE:", mse_f)
    print("Fixed-point MSE:", mse_q)

    # Parameter errors
    w_abs_err = np.abs(w_q - w_f)
    b_abs_err = abs(b_q - b_f)
    print("Param abs error |w|:", w_abs_err)
    print("Param abs error |b|:", b_abs_err)

    # Prediction errors
    pred_abs_err = np.abs(preds_q - preds_f)
    pred_rel_err = pred_abs_err / (np.abs(preds_f) + 1e-18)
    print("Per-test absolute prediction error:", pred_abs_err)
    print("Per-test relative prediction error:", pred_rel_err)
    print("RMSE between predictions:", float(np.sqrt(np.mean((preds_q - preds_f) ** 2))))

    # Preserve original assertion and add a parallel one for fixed-point
    assert mse_f <= 1.0
    assert mse_q <= 1.0


if __name__ == "__main__":
    with open('../mlalgo/linear_regression/sol.rs.in', 'r') as f:
        input_data = json.load(f)

    training_x = np.asarray(input_data['training_x']).reshape((10, 2))
    training_y = np.asarray(input_data['training_y'])
    testing_x = np.asarray(input_data['testing_x']).reshape((2, 2))
    testing_y = np.asarray(input_data['testing_y'])

    verify_solution(training_x, training_y, testing_x, testing_y)


# RMSE between predictions: 1.2560739669470201e-15
