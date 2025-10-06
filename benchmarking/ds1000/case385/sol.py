import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 4, 4], result: NDArray[int, 4, 2, 2]):
    # a =
    # [[1,  5,  9, 13],
    #  [2,  6, 10, 14],
    #  [3,  7, 11, 15],
    #  [4,  8, 12, 16]]
    #
    # Expected result =
    # [
    #   [[1, 5],
    #    [2, 6]],
    #   [[3, 7],
    #    [4, 8]],
    #   [[9, 13],
    #    [10, 14]],
    #   [[11, 15],
    #    [12, 16]]
    # ]

    # Step 1: reshape to 4D (2, 2, 2, 2)
    reshaped = a.reshape((2, 2, 2, 2))

    # Step 2: transpose to reorder patch grouping (0, 2, 1, 3)
    transposed = reshaped.transpose((0, 2, 1, 3)).transpose((1, 0, 2, 3))

    # Step 3: reshape to final (4, 2, 2)
    computed = transposed.reshape((4, 2, 2))

    assert result == computed


if __name__ == '__main__':
    a = [
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
        [4, 8, 12, 16],
    ]
    result = [
        [[1, 5], [2, 6]],
        [[3, 7], [4, 8]],
        [[9, 13], [10, 14]],
        [[11, 15], [12, 16]],
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
