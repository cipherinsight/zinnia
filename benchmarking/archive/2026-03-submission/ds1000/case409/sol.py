import json

from zinnia import *


@zk_circuit
def verify_solution(x: NDArray[int, 3, 3], y: NDArray[int, 3, 3], z: NDArray[int, 3, 3]):
    # x =
    # [[2, 2, 2],
    #  [2, 2, 2],
    #  [2, 2, 2]]
    # y =
    # [[3, 3, 3],
    #  [3, 3, 3],
    #  [3, 3, 1]]
    # Expected:
    # z = x + y =
    # [[5, 5, 5],
    #  [5, 5, 5],
    #  [5, 5, 3]]

    expected = x + y
    assert z == expected


if __name__ == '__main__':
    x = [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2]
    ]
    y = [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 1]
    ]
    z = [
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 3]
    ]

    assert verify_solution(x, y, z)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(x, y, z)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
