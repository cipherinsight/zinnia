import json
from typing import Tuple

from zinnia import NDArray, np

# ---------------------------
# Fixed-point configuration
# ---------------------------
FXP_FRAC_BITS = 64
FXP_ONE = 1 << FXP_FRAC_BITS
FXP_HALF = 1 << (FXP_FRAC_BITS - 1)


np.random.seed(42)


def to_fxp(x: np.ndarray) -> np.ndarray:
    """
    Quantize float64 -> signed integer fixed point with scale 2^64.
    Uses round-to-nearest (ties to away-from-zero).
    """
    # Use Python ints (dtype=object) so we don't overflow.
    # If you prefer, you can try int128 with numpy if available.
    scaled = np.rint(x * FXP_ONE).astype(object)
    return scaled


def from_fxp(q: np.ndarray) -> np.ndarray:
    """Dequantize fixed-point -> float64."""
    return np.asarray(q, dtype=float) / FXP_ONE


def fxp_add(a: int, b: int) -> int:
    return a + b


def fxp_sub(a: int, b: int) -> int:
    return a - b


def fxp_mul(a: int, b: int) -> int:
    """
    (a * b) / 2^64 with round-to-nearest. Works for signed ints.
    """
    prod = a * b
    if prod >= 0:
        return (prod + FXP_HALF) // FXP_ONE
    else:
        return (prod - FXP_HALF) // FXP_ONE


def fxp_sq(a: int) -> int:
    # a^2 is non-negative; rounding is straightforward.
    prod = a * a
    return (prod + FXP_HALF) // FXP_ONE


def fxp_div_by_int(a: int, k: int) -> int:
    """
    Divide fixed-point value `a` by *integer* k, preserving fixed-point scale.
    Round-to-nearest (ties away-from-zero).
    """
    if a >= 0:
        return (a + k // 2) // k
    else:
        return (a - k // 2) // k


def kmeans_float64(
    data: NDArray[float, 10, 2],
    centroids: NDArray[float, 3, 2],
    iters: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = data.shape
    classes = centroids.shape[0]
    labels = np.zeros((n,), dtype=int)
    c = centroids.copy()
    for _ in range(iters):
        # assign
        for i in range(n):
            dists = np.zeros((classes,), dtype=float)
            for j in range(classes):
                diff = data[i] - c[j]
                dist = diff[0] * diff[0] + diff[1] * diff[1]
                dists[j] = dist
            labels[i] = int(np.argmin(dists))
        # update
        new_c = np.zeros((classes, d), dtype=float)
        counts = np.zeros((classes,), dtype=float)
        for i in range(n):
            new_c[labels[i]] += data[i]
            counts[labels[i]] += 1.0
        for j in range(classes):
            new_c[j] /= counts[j]
        c = new_c
    return c, labels


def kmeans_fixed_point(
    data: NDArray[float, 10, 2],
    centroids: NDArray[float, 3, 2],
    iters: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same algorithm as float path, but all arithmetic is performed in 64-bit
    fixed-point represented as Python ints. Returns (centroids_float, labels).
    """
    n, d = data.shape
    classes = centroids.shape[0]

    # Quantize inputs
    data_q = to_fxp(np.asarray(data, dtype=float))
    c_q = to_fxp(np.asarray(centroids, dtype=float))

    labels = np.zeros((n,), dtype=int)

    for _ in range(iters):
        # assign
        for i in range(n):
            # store distances in fixed-point too (they are >= 0)
            dists_q = [0] * classes
            for j in range(classes):
                # dist = (x0-c0)^2 + (x1-c1)^2 in fixed point
                dx0 = fxp_sub(data_q[i][0], c_q[j][0])
                dx1 = fxp_sub(data_q[i][1], c_q[j][1])
                dist_q = fxp_sq(dx0)
                dist_q = fxp_add(dist_q, fxp_sq(dx1))
                dists_q[j] = dist_q
            labels[i] = int(np.argmin(dists_q))

        # update
        new_c_q = np.zeros((classes, d), dtype=object)
        counts = np.zeros((classes,), dtype=int)
        for i in range(n):
            lbl = labels[i]
            new_c_q[lbl][0] = fxp_add(int(new_c_q[lbl][0]), int(data_q[i][0]))
            new_c_q[lbl][1] = fxp_add(int(new_c_q[lbl][1]), int(data_q[i][1]))
            counts[lbl] += 1

        for j in range(classes):
            k = counts[j]
            # average: divide by integer k in fixed-point domain
            new_c_q[j][0] = fxp_div_by_int(int(new_c_q[j][0]), int(k))
            new_c_q[j][1] = fxp_div_by_int(int(new_c_q[j][1]), int(k))

        c_q = new_c_q

    # Dequantize centroids back to float64 for comparison
    c_float = from_fxp(c_q)
    return c_float, labels


def verify_solution(
    data: NDArray[float, 10, 2],
    centroids: NDArray[float, 3, 2],
    classifications: NDArray[int, 10],
):
    # Run float64 baseline
    c_f64, labels_f64 = kmeans_float64(data, centroids, iters=10)

    # Run fixed-point (2^-64) version
    c_fxp, labels_fxp = kmeans_fixed_point(data, centroids, iters=10)

    # Report final centroids
    print("Final centroids (float64):")
    print(c_f64)
    print("Final centroids (fixed-point->float):")
    print(c_fxp)

    # Error metrics
    diff = c_fxp - c_f64
    abs_err = np.abs(diff)
    rel_err = abs_err / (np.abs(c_f64) + 1e-18)
    rmse = np.sqrt(np.mean(diff ** 2))

    print("\nPer-coordinate absolute error:")
    print(abs_err)
    print("\nPer-coordinate relative error:")
    print(rel_err)
    print(f"\nRMSE over all centroid coordinates: {rmse:.6e}")

    # Label comparisons
    label_match_f64 = np.array_equal(labels_f64, classifications)
    label_match_fxp = np.array_equal(labels_fxp, classifications)
    cross_match = np.array_equal(labels_fxp, labels_f64)
    mismatches = int(np.sum(labels_fxp != labels_f64))

    print("\nLabel checks:")
    print(f"- float64 labels match provided classifications: {label_match_f64}")
    print(f"- fixed-point labels match provided classifications: {label_match_fxp}")
    print(f"- fixed-point vs float64 labels identical: {cross_match} "
          f"(mismatches: {mismatches})")


if __name__ == "__main__":
    with open('../mlalgo/kmeans/sol.rs.in', 'r') as f:
        input_data = json.load(f)

    data = np.random.normal(size=(100, 2))
    cents = np.random.normal(size=(3, 2))
    classes = np.asarray(input_data['classifications']).repeat(10)

    verify_solution(data, cents, classes)


# RMSE over all centroid coordinates: 1.281975e-16