import json
from typing import Tuple
from zinnia import NDArray, np

np.random.seed(42)

# ---------------------------
# Fixed-point configuration
# ---------------------------
FXP_FRAC_BITS = 64
FXP_ONE = 1 << FXP_FRAC_BITS
FXP_HALF = 1 << (FXP_FRAC_BITS - 1)

def to_fxp(arr: np.ndarray) -> np.ndarray:
    """Quantize float64 -> signed integer fixed point (scale 2^64, round-to-nearest)."""
    return np.rint(arr.astype(float) * FXP_ONE).astype(object)

def from_fxp(arr_q: np.ndarray) -> np.ndarray:
    """Dequantize fixed-point -> float64."""
    return np.asarray(arr_q, dtype=float) / FXP_ONE

def fxp_add(a: int, b: int) -> int:
    return a + b

def fxp_sub(a: int, b: int) -> int:
    return a - b

def fxp_mul(a: int, b: int) -> int:
    """(a * b) / 2^64 with round-to-nearest (sign-aware)."""
    prod = a * b
    if prod >= 0:
        return (prod + FXP_HALF) // FXP_ONE
    else:
        return (prod - FXP_HALF) // FXP_ONE

def fxp_dot(vec_a_q: np.ndarray, vec_b_q: np.ndarray) -> int:
    """Dot product in fixed-point (returns fixed-point)."""
    s = 0
    for ai, bi in zip(vec_a_q, vec_b_q):
        s = fxp_add(s, fxp_mul(int(ai), int(bi)))
    return s

def perceptron_train_float(
    X: NDArray[float, 10, 2],
    y: NDArray[int, 10],
    w0: NDArray[float, 2],
    epochs: int = 50,
) -> np.ndarray:
    n, d = X.shape
    w = np.asarray(w0, dtype=float).copy()
    for _ in range(epochs):
        for i in range(n):
            activation = float(np.dot(w, X[i]))
            pred = 1 if activation >= -1e-10 else -1
            if pred != int(y[i]):
                # Update: w += y_i * x_i  where y_i in {+1,-1}
                w += X[i] if int(y[i]) == 1 else -X[i]
    return w

def perceptron_eval_float(
    Xtest: NDArray[float, 2, 2], ytest: NDArray[int, 2], w: np.ndarray
) -> Tuple[np.ndarray, float]:
    preds = np.zeros((len(ytest),), dtype=int)
    correct = 0
    for i in range(len(ytest)):
        activation = float(np.dot(w, Xtest[i]))
        preds[i] = 1 if activation >= 0.0 else -1
        if preds[i] == int(ytest[i]):
            correct += 1
    acc = correct / float(len(ytest))
    return preds, acc

def perceptron_train_fixed_point(
    X: NDArray[float, 10, 2],
    y: NDArray[int, 10],
    w0: NDArray[float, 2],
    epochs: int = 50,
) -> np.ndarray:
    """
    Train perceptron entirely in fixed-point integer arithmetic.
    Returns final weights dequantized to float64 for reporting/comparison.
    """
    n, d = X.shape
    Xq = to_fxp(np.asarray(X, dtype=float))
    wq = to_fxp(np.asarray(w0, dtype=float))
    y_int = np.asarray(y, dtype=int)
    for _ in range(epochs):
        for i in range(n):
            act_q = fxp_dot(wq, Xq[i])              # fixed-point value
            pred = 1 if from_fxp(act_q) >= -1e-10 else -1         # compare in integer domain
            if pred != int(y_int[i]):
                # wq += y_i * x_i  (all fixed-point integers)
                if int(y_int[i]) == 1:
                    wq[0] = fxp_add(int(wq[0]), int(Xq[i][0]))
                    wq[1] = fxp_add(int(wq[1]), int(Xq[i][1]))
                else:
                    wq[0] = fxp_sub(int(wq[0]), int(Xq[i][0]))
                    wq[1] = fxp_sub(int(wq[1]), int(Xq[i][1]))
    return from_fxp(wq)


def perceptron_eval_fixed_point(
    Xtest: NDArray[float, 2, 2], ytest: NDArray[int, 2], w_float_from_q: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Evaluate by quantizing X_test and w, doing integer dot+sign, and return preds+accuracy.
    """
    Xtest_q = to_fxp(np.asarray(Xtest, dtype=float))
    wq = to_fxp(np.asarray(w_float_from_q, dtype=float))
    preds = np.zeros((len(ytest),), dtype=int)
    correct = 0
    for i in range(len(ytest)):
        act_q = fxp_dot(wq, Xtest_q[i])
        preds[i] = 1 if act_q >= 0 else -1
        if preds[i] == int(ytest[i]):
            correct += 1
    acc = correct / float(len(ytest))
    return preds, acc

def verify_solution(
        training_data: NDArray[float, 10, 2],
        training_labels: NDArray[int, 10],
        initial_weights: NDArray[float, 2],
        testing_data: NDArray[float, 2, 2],
        testing_labels: NDArray[int, 2],
):
    # 1) Train/eval in float64
    w_f = perceptron_train_float(training_data, training_labels, initial_weights, epochs=50)
    preds_f, acc_f = perceptron_eval_float(testing_data, testing_labels, w_f)

    # 2) Train/eval in 2^-64 fixed-point (integer arithmetic)
    w_q_as_float = perceptron_train_fixed_point(training_data, training_labels, initial_weights, epochs=50)
    preds_q, acc_q = perceptron_eval_fixed_point(testing_data, testing_labels, w_q_as_float)

    # 3) Report differences
    print("Final weights (float64):", w_f)
    print("Final weights (fixed-point→float):", w_q_as_float)
    print("Abs weight error:", np.abs(w_q_as_float - w_f))
    print("RMSE on weights:", np.sqrt(np.mean((w_q_as_float - w_f) ** 2)))
    print("Test preds (float64):", preds_f)
    print("Test preds (fixed-point):", preds_q)
    print(f"Accuracies — float64: {acc_f:.3f}, fixed-point: {acc_q:.3f}")
    print("Prediction mismatch count:", int(np.sum(preds_f != preds_q)))


if __name__ == "__main__":
    with open('../mlalgo/neuron/sol.rs.in', 'r') as f:
        input_data = json.load(f)

    training_data = np.random.normal(size=(10, 2))
    training_labels = np.random.normal(size=(10, ))
    initial_weights = np.random.normal(size=(2, ))
    testing_data = np.random.normal(size=(2, 2))
    testing_labels = np.random.normal(size=(2,))

    verify_solution(training_data, training_labels, initial_weights, testing_data, testing_labels)

# RMSE on weights: 5.025300626495813e-13
