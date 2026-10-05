import json

from zinnia import *


@zk_circuit
def verify_solution(A: DynamicNDArray[int, 7, 1], B: DynamicNDArray[int, 6, 2]):
    flat_b = B.reshape((6,))
    n = flat_b.shape[0]
    for i in range(n):
        assert flat_b[i] == A[n - i]



if __name__ == '__main__':
    A = [1, 2, 3, 4, 5, 6, 7]
    B = [
        [7, 6],
        [5, 4],
        [3, 2]
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
