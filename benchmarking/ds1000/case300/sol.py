import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[float, 5], p: float, result: float):
    # a = [1, 2, 3, 4, 5], p = 25
    n = 5

    # Verify that the array is sorted in non-decreasing order
    for i in range(n - 1):
        assert a[i] <= a[i + 1]

    # Compute percentile rank
    rank = (p / 100.0) * (n - 1)
    lower = int(rank)
    upper = lower + 1
    fraction = rank - lower

    # Linear interpolation between lower and upper elements
    interpolated = a[lower] + (a[upper] - a[lower]) * fraction

    assert result == interpolated




if __name__ == '__main__':
    a = [1, 2, 3, 4, 5]
    p = 25
    result = 2.0

    assert verify_solution(a, p, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, p, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
