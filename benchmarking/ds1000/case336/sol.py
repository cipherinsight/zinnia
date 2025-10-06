import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 5, 5], result: NDArray[int, 5]):
    # a =
    # [[ 0,  1,  2,  3,  4],
    #  [ 5,  6,  7,  8,  9],
    #  [10, 11, 12, 13, 14],
    #  [15, 16, 17, 18, 19],
    #  [20, 21, 22, 23, 24]]
    #
    # Expected result = [4, 8, 12, 16, 20]
    # Reference: result = np.diag(np.fliplr(a))
    #
    # Since diag and fliplr are not available, we manually reverse columns and collect the diagonal.

    flipped = np.zeros(a.shape, dtype=int)
    for i in range(5):
        for j in range(5):
            flipped[i, j] = a[i, 4 - j]  # horizontally flip (fliplr)

    diag_vals = np.zeros((5, ), dtype=int)
    for k in range(5):
        diag_vals[k] = flipped[k, k]  # extract diagonal

    assert result == diag_vals


if __name__ == '__main__':
    a = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]
    ]
    result = [4, 8, 12, 16, 20]
    assert verify_solution(a, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
