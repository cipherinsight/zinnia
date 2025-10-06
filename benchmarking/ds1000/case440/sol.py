import json

from zinnia import *


@zk_circuit
def verify_solution(Y: NDArray[int, 4, 3, 3], X: NDArray[float, 3, 4]):
    # Y has shape (N=4, M=3, M=3), where Y[i] = x_i @ x_i^T and all entries of X are positive.
    # Recover X by taking the square root of the diagonal of each MxM slice Y[i].
    # For this concrete instance, the recovered X (M x N) should be:
    # Columns (i = 0..3):
    #  i=0: sqrt(diag([[81,63,63],[63,49,49],[63,49,49]])) -> [9,7,7]
    #  i=1: sqrt(diag([[ 4,12, 8],[12,36,24],[ 8,24,16]])) -> [2,6,4]
    #  i=2: sqrt(diag([[25,35,25],[35,49,35],[25,35,25]])) -> [5,7,5]
    #  i=3: sqrt(diag([[25,30,10],[30,36,12],[10,12, 4]])) -> [5,6,2]
    # So X =
    # [[9, 2, 5, 5],
    #  [7, 6, 7, 6],
    #  [7, 4, 5, 2]]

    M = X.shape[0]
    N = X.shape[1]

    expected = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(N):        # column index in X
        for j in range(M):    # row index in X
            # Diagonal entry of Y[i] at (j,j) equals X[j,i]^2
            expected[j][i] = (Y[i, j, j]) ** 0.5

    assert X == expected


if __name__ == '__main__':
    Y = [
        [[81, 63, 63],
         [63, 49, 49],
         [63, 49, 49]],

        [[4, 12, 8],
         [12, 36, 24],
         [8, 24, 16]],

        [[25, 35, 25],
         [35, 49, 35],
         [25, 35, 25]],

        [[25, 30, 10],
         [30, 36, 12],
         [10, 12, 4]],
    ]
    X = [
        [9.0, 2.0, 5.0, 5.0],
        [7.0, 6.0, 7.0, 6.0],
        [7.0, 4.0, 5.0, 2.0],
    ]

    assert verify_solution(Y, X)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(Y, X)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
