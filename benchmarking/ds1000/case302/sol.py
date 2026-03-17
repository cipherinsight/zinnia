import json

from zinnia import *


@zk_circuit
def verify_solution(A: DynamicNDArray[int, 6, 1], B: DynamicNDArray[int, 6, 2]):
    nrow = 3
    ncol = A.shape[0] // nrow
    for r in range(nrow):
        for c in range(ncol):
            idx = r * ncol + c
            assert B[idx] == A[idx]




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
