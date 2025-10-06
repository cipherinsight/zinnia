import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 3, 2], mask: NDArray[int, 3, 2]):
    # a =
    # [[0, 1],
    #  [2, 1],
    #  [4, 8]]
    # For each row, find the minimum and mark True where the element equals the row min.
    # Expected mask =
    # [[True, False],
    #  [False, True],
    #  [True, False]]

    # Since keepdims is unavailable, emulate by reshaping
    row_min = a.min(axis=1).reshape((a.shape[0], 1))
    expected = (a == row_min)
    assert (mask == 1) == expected


if __name__ == '__main__':
    a = [
        [0, 1],
        [2, 1],
        [4, 8]
    ]
    mask = [
        [1, 0],
        [0, 1],
        [1, 0]
    ]

    assert verify_solution(a, mask)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, mask)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
