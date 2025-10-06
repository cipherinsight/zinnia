import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 5, 5], result: NDArray[int, 2, 5]):
    # a =
    # [[ 0,  1,  2,  3,  4],
    #  [ 5,  6,  7,  8,  9],
    #  [10, 11, 12, 13, 14],
    #  [15, 16, 17, 18, 19],
    #  [20, 21, 22, 23, 24]]
    #
    # Expected result =
    # [[ 0,  6, 12, 18, 24],
    #  [ 4,  8, 12, 16, 20]]
    #
    # Reference: result = np.vstack((np.diag(a), np.diag(np.fliplr(a))))
    #
    # Since diag and fliplr are not available, we manually compute both diagonals.

    n = a.shape[0]
    main_diag = np.zeros((n, ), dtype=int)
    for i in range(n):
        main_diag[i] = a[i, i]

    flipped = np.zeros(a.shape, dtype=int)
    for i in range(n):
        for j in range(n):
            flipped[i, j] = a[i, (n - 1) - j]

    anti_diag = np.zeros((n, ), dtype=int)
    for i in range(n):
        anti_diag[i] = flipped[i, i]

    stacked = np.zeros((2, n), dtype=int)
    stacked[0, :] = main_diag
    stacked[1, :] = anti_diag

    assert result == stacked


if __name__ == '__main__':
    a = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]
    ]
    result = [
        [0, 6, 12, 18, 24],
        [4, 8, 12, 16, 20]
    ]
    assert verify_solution(a, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
