import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 5, 6], result: NDArray[int, 2, 5]):
    # a =
    # [[ 0,  1,  2,  3,  4,  5],
    #  [ 5,  6,  7,  8,  9, 10],
    #  [10, 11, 12, 13, 14, 15],
    #  [15, 16, 17, 18, 19, 20],
    #  [20, 21, 22, 23, 24, 25]]
    #
    # Reference behavior:
    # dim = min(a.shape) = 5
    # b = a[:dim, :dim]
    # result = np.vstack((np.diag(b), np.diag(np.fliplr(b))))
    #
    # Expected:
    # [[ 0,  6, 12, 18, 24],
    #  [ 4,  8, 12, 16, 20]]

    nrows, ncols = 5, 6
    dim = min(nrows, ncols)

    # Extract the square submatrix b = a[:dim, :dim]
    b = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            b[i, j] = a[i, j]

    # Compute the main diagonal
    main_diag = np.zeros((dim, ), dtype=int)
    for i in range(dim):
        main_diag[i] = b[i, i]

    # Compute the horizontally flipped matrix
    flipped = np.zeros(b.shape, dtype=int)
    for i in range(dim):
        for j in range(dim):
            flipped[i, j] = b[i, (dim - 1) - j]

    # Compute the anti-diagonal
    anti_diag = np.zeros((dim, ), dtype=int)
    for i in range(dim):
        anti_diag[i] = flipped[i, i]

    # Stack them together
    stacked = np.zeros((2, dim), dtype=int)
    stacked[0, :] = main_diag
    stacked[1, :] = anti_diag

    assert result == stacked


if __name__ == '__main__':
    a = [
        [0, 1, 2, 3, 4, 5],
        [5, 6, 7, 8, 9, 10],
        [10, 11, 12, 13, 14, 15],
        [15, 16, 17, 18, 19, 20],
        [20, 21, 22, 23, 24, 25]
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
