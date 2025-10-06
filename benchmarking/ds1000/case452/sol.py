import json

from zinnia import *


@zk_circuit
def verify_solution(X: NDArray[int, 5, 4], result: NDArray[float, 5, 4]):
    # X =
    # [[ 1, -2,  3,   6],
    #  [ 4,  5, -6,   5],
    #  [-1,  2,  5,   5],
    #  [ 4,  5, 10, -25],
    #  [ 5, -2, 10,  25]]
    #
    # Each row normalized by its L1 norm:
    # l1 = sum(abs(X), axis=1)
    # result = X / l1.reshape(-1, 1)

    rows = X.shape[0]
    cols = X.shape[1]

    l1 = []
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            val = X[i, j]
            s += val if val >= 0 else -val
        l1.append(s)

    expected = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(X[i, j] / l1[i])
        expected.append(row)

    assert result == expected


if __name__ == '__main__':
    X = [
        [1, -2, 3, 6],
        [4, 5, -6, 5],
        [-1, 2, 5, 5],
        [4, 5, 10, -25],
        [5, -2, 10, 25]
    ]

    import numpy as np
    l1 = np.abs(X).sum(axis=1)
    result = (np.array(X) / l1.reshape(-1, 1)).tolist()

    assert verify_solution(X, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(X, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
