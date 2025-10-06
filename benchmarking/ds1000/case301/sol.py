import json

from zinnia import *


@zk_circuit
def verify_solution(A: NDArray[int, 6], B: NDArray[int, 3, 2]):
    # A = [1, 2, 3, 4, 5, 6], ncol = 2
    # Expected reshape result:
    # [[1, 2],
    #  [3, 4],
    #  [5, 6]]

    # Verify reshape correctness
    nrow = 3
    ncol = 2
    for i in range(nrow):
        for j in range(ncol):
            # Compute corresponding 1D index
            idx = i * ncol + j
            assert B[i, j] == A[idx]




if __name__ == '__main__':
    A = [1, 2, 3, 4, 5, 6]
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
