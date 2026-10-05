import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 4, 4], result: NDArray[int, 4, 4]):
    # a =
    # [[0, 3, 1, 3],
    #  [3, 0, 0, 0],
    #  [1, 0, 0, 0],
    #  [3, 0, 0, 0]]
    # zero_rows = [1, 3]
    # zero_cols = [1, 2]
    # Expected:
    # [[0, 0, 0, 3],
    #  [0, 0, 0, 0],
    #  [1, 0, 0, 0],
    #  [0, 0, 0, 0]]

    zero_rows = [1, 3]
    zero_cols = [1, 2]

    modified = a
    for r in zero_rows:
        modified[r, :] = 0
    for c in zero_cols:
        modified[:, c] = 0
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
        [0, 0, 0, 3],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
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
