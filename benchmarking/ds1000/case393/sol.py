import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 24, 2], result: DynamicNDArray[int, 21, 2]):
    rows = 3
    cols = a.shape[0] // rows
    trimmed_cols = cols - 1

    expected_flat = np.concatenate((a[1:cols], a[(cols + 1):(2 * cols)], a[(2 * cols + 1):(3 * cols)]), axis=0)
    expected = expected_flat.reshape((rows, trimmed_cols)).reshape((result.shape[0],))
    assert result.reshape((result.shape[0],)) == expected


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
