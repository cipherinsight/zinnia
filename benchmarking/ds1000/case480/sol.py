import json

from zinnia import *


@zk_circuit
def verify_solution(x: NDArray[int, 9], y: NDArray[int, 9], a: int, b: int, result: int):
    # x = [0, 1, 1, 1, 3, 1, 5, 5, 5]
    # y = [0, 2, 3, 4, 2, 4, 3, 4, 5]
    # (a, b) = (1, 4)
    # We search for the first index i such that x[i] == a and y[i] == b.
    # Expected result: 3

    n = 9
    found_index = -1
    for i in range(n):
        if (x[i] == a) and (y[i] == b) and (found_index == -1):
            found_index = i

    expected = found_index
    assert result == expected


if __name__ == '__main__':
    x = [0, 1, 1, 1, 3, 1, 5, 5, 5]
    y = [0, 2, 3, 4, 2, 4, 3, 4, 5]
    a = 1
    b = 4
    result = 3

    assert verify_solution(x, y, a, b, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(x, y, a, b, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
