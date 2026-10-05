import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 16, 2], result: DynamicNDArray[int, 16, 3]):
    rows = a.shape[0] // 4
    cols = a.shape[0] // rows

    row_pairs = rows // 2
    col_pairs = cols // 2

    expected = a.reshape((row_pairs, 2, col_pairs, 2)).transpose((2, 0, 1, 3)).reshape((result.shape[0],))
    assert result.reshape((result.shape[0],)) == expected


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
    # assert verify_solution(a, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
