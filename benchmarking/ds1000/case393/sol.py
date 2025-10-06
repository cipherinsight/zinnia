import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 3, 8], result: NDArray[int, 3, 7]):
    # a =
    # [[ 0,  1,  2,  3, 5, 6, 7, 8],
    #  [ 4,  5,  6,  7, 5, 3, 2, 5],
    #  [ 8,  9, 10, 11, 4, 5, 3, 5]]
    # Extract columns in range [low, min(high, a.shape[1])) with low=1, high=10
    low = 1
    # a.shape[1] == 8, so clamped high = 8
    clamped_high = 8
    expected = a[:, low:clamped_high]
    assert result == expected


if __name__ == '__main__':
    a = [
        [0, 1, 2, 3, 5, 6, 7, 8],
        [4, 5, 6, 7, 5, 3, 2, 5],
        [8, 9, 10, 11, 4, 5, 3, 5]
    ]
    result = [
        [1, 2, 3, 5, 6, 7, 8],
        [5, 6, 7, 5, 3, 2, 5],
        [9, 10, 11, 4, 5, 3, 5]
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
