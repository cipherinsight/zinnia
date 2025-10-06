import json

from zinnia import *


@zk_circuit
def verify_solution(grades: NDArray[float, 27], result: NDArray[float, 27]):
    # Goal: emulate R's ecdf(x)(x): return ECDF values for x in increasing order.
    # Zinnia has no sort; so we VERIFY 'grades' is already sorted non-decreasingly.
    n = 27

    # 1) Validate sortedness (non-decreasing)
    for i in range(n - 1):
        assert grades[i] <= grades[i + 1]

    # 2) ECDF values at sorted sample points: i/n for i=1..n
    ys = np.zeros((n, ), dtype=int)
    for i in range(n):
        ys[i] = (i + 1) / float(n)

    # 3) Verify output
    assert result == ys


if __name__ == '__main__':
    grades = [
        60.8, 61.0, 65.5, 69.0, 76.0, 76.0, 78.0, 78.0, 82.0,
        86.0, 87.5, 89.5, 91.0, 91.5, 92.3, 92.5, 92.8, 93.0,
        93.5, 93.5, 94.5, 94.5, 95.0, 95.5, 98.0, 98.5, 99.5
    ]
    result = [i / 27.0 for i in range(1, 28)]

    assert verify_solution(grades, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(grades, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
