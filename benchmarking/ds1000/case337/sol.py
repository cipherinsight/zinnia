import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 5, 6], result: NDArray[int, 5]):
    # Expected: [5, 9, 13, 17, 21]
    nrows = a.shape[0]
    ncols = a.shape[1]

    flipped = np.zeros(a.shape, dtype=int)
    for i in range(nrows):
        for j in range(ncols):
            flipped[i, j] = a[i, (ncols - 1) - j]  # correct fliplr

    diag_vals = np.zeros((nrows, ), dtype=int)
    for k in range(nrows):
        diag_vals[k] = flipped[k, k]

    assert result == diag_vals


if __name__ == '__main__':
    a = [
        [0, 1, 2, 3, 4, 5],
        [5, 6, 7, 8, 9, 10],
        [10, 11, 12, 13, 14, 15],
        [15, 16, 17, 18, 19, 20],
        [20, 21, 22, 23, 24, 25]
    ]
    result = [5, 9, 13, 17, 21]
    assert verify_solution(a, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
