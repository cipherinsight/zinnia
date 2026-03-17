import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 20, 2], result: DynamicNDArray[int, 16, 3]):
    rows = 4
    cols = a.shape[0] // rows
    kept_cols = cols - 1

    trimmed_flat = np.concatenate((a[0:kept_cols], a[cols:(cols + kept_cols)], a[2 * cols:(2 * cols + kept_cols)], a[3 * cols:(3 * cols + kept_cols)]), axis=0)
    trimmed = trimmed_flat.reshape((rows, kept_cols))

    row_pairs = rows // 2
    col_pairs = kept_cols // 2
    expected = trimmed.reshape((row_pairs, 2, col_pairs, 2)).transpose((0, 2, 1, 3)).reshape((result.shape[0],))
    assert result.reshape((result.shape[0],)) == expected


if __name__ == '__main__':
    a = [
        [1, 5, 9, 13, 17],
        [2, 6, 10, 14, 18],
        [3, 7, 11, 15, 19],
        [4, 8, 12, 16, 20],
    ]
    result = [
        [[1, 5], [2, 6]],
        [[9, 13], [10, 14]],
        [[3, 7], [4, 8]],
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
