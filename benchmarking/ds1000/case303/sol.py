import json

from zinnia import *


@zk_circuit
def verify_solution(A: NDArray[int, 7], B: NDArray[int, 3, 2]):
    # A = [1, 2, 3, 4, 5, 6, 7], ncol = 2
    # Truncate last element -> [1, 2, 3, 4, 5, 6]
    # Expected reshape result:
    # [[1, 2],
    #  [3, 4],
    #  [5, 6]]

    ncol = 2
    nrow = 3   # since (7 // 2) = 3 full rows
    truncated = [A[0], A[1], A[2], A[3], A[4], A[5]]

    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            assert B[i, j] == truncated[idx]



if __name__ == '__main__':
    A = [1, 2, 3, 4, 5, 6, 7]
    B = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]

    assert verify_solution(A, B)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(A, B)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
