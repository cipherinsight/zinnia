import json

from zinnia import *


@zk_circuit
def verify_solution(grades: NDArray[float, 27], threshold: float, low: float, high: float):
    # Goal: ECDF over 'grades' (sorted ascending, verified in-circuit), then
    # find the longest interval [low, high) such that ECDF(x) < threshold for any x in [low, high),
    # with low, high constrained to be elements of the original (sorted) array.
    #
    # Reference logic:
    # xs = sort(grades); ys = (1..n)/n; t = argmax(ys > threshold)
    # low = xs[0]; high = xs[t]
    #
    # We don't sort in-circuit; instead, we VERIFY that 'grades' is already sorted.

    n = 27

    # 1) Verify sortedness (non-decreasing)
    for i in range(n - 1):
        assert grades[i] <= grades[i + 1]

    # 2) Compute the first index t where ECDF exceeds threshold:
    #    ECDF at grades[k] is (k+1)/n, so find the smallest k with (k+1)/n > threshold.
    t = n  # sentinel; will be updated once
    for k in range(n):
        cond = ( (k + 1) / float(n) ) > threshold
        # emulate argmax of boolean vector (first True)
        if cond and (t == n):
            t = k

    # 3) Determine low, high as elements in the array
    computed_low = grades[0]
    computed_high = grades[t]  # by construction t in [0, n-1] since threshold < 1 for typical ECDF query

    # 4) Verify outputs
    assert low == computed_low
    assert high == computed_high


if __name__ == '__main__':
    # Provide 'grades' already sorted (since circuit verifies sortedness and does not sort).
    grades = [
        60.8, 61.0, 65.5, 69.0, 76.0, 76.0, 78.0, 78.0, 82.0,
        86.0, 87.5, 89.5, 91.0, 91.5, 92.3, 92.5, 92.8, 93.0,
        93.5, 93.5, 94.5, 94.5, 95.0, 95.5, 98.0, 98.5, 99.5
    ]
    threshold = 0.5
    # n = 27; first index with (k+1)/27 > 0.5 is k = 13  -> high = grades[13] = 91.5
    low = 60.8
    high = 91.5

    assert verify_solution(grades, threshold, low, high)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(grades, threshold, low, high)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
