import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[float, 13], result: Tuple[float, float]):
    # a = [0, 1, 2, 5, 6, 7, 8, 8, 8, 10, 29, 32, 45]
    # Compute mean and standard deviation manually, since np.std is unavailable.
    # 3σ interval = (μ - 3σ, μ + 3σ)

    n = 13
    mean_val = np.mean(a)
    # manual std: sqrt(sum((x - mean)^2) / n)
    variance = np.sum((a - mean_val) * (a - mean_val))
    variance /= n
    std_val = variance ** 0.5

    lower = mean_val - 3 * std_val
    upper = mean_val + 3 * std_val
    expected = (lower, upper)
    assert result == expected


if __name__ == '__main__':
    a = [0, 1, 2, 5, 6, 7, 8, 8, 8, 10, 29, 32, 45]
    # compute expected result
    import numpy as np
    mean_val = np.mean(a)
    std_val = (sum([(x - mean_val) ** 2 for x in a]) / len(a)) ** 0.5
    result = (mean_val - 3 * std_val, mean_val + 3 * std_val)

    assert verify_solution(a, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
