import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 4, 4], result: NDArray[int, 4, 4]):
    # a =
    # [[0, 3, 1, 3],
    #  [3, 0, 0, 0],
    #  [1, 0, 0, 0],
    #  [3, 0, 0, 0]]
    # zero_rows = 0, zero_cols = 0
    # Expected:
    # [[0, 0, 0, 0],
    #  [0, 0, 0, 0],
    #  [0, 0, 0, 0],
    #  [0, 0, 0, 0]]

    zero_rows = 0
    zero_cols = 0

    modified = a
    modified[zero_rows, :] = 0
    modified[:, zero_cols] = 0
    expected = modified
    assert result == expected


if __name__ == '__main__':
    a = [
        [0, 3, 1, 3],
        [3, 0, 0, 0],
        [1, 0, 0, 0],
        [3, 0, 0, 0]
    ]
    result = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    assert verify_solution(a, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
