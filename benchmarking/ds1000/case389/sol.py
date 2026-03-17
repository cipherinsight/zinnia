import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 24, 3], result: DynamicNDArray[int, 24, 2]):
    block_rows = 2
    block_cols = 2
    row_size = a.shape[0] // (block_rows * block_cols * 2)

    expected = a.reshape((block_rows, 2, block_cols, row_size)).transpose((0, 2, 1, 3)).reshape((result.shape[0],))
    assert result.reshape((result.shape[0],)) == expected


if __name__ == '__main__':
    a = [
        [[0, 1, 2],
         [6, 7, 8]],

        [[3, 4, 5],
         [9, 10, 11]],

        [[12, 13, 14],
         [18, 19, 20]],

        [[15, 16, 17],
         [21, 22, 23]],
    ]
    result = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
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
