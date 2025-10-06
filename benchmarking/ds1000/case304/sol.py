import json

from zinnia import *


@zk_circuit
def verify_solution(A: NDArray[int, 7], B: NDArray[int, 3, 2]):
    # A = [1,2,3,4,5,6,7], ncol = 2
    # Keep last 6 elements, reverse them, then reshape (3,2)
    # Reference behavior:
    #   truncated = A[1:]      -> [2,3,4,5,6,7]
    #   reversed_part = truncated[::-1] -> [7,6,5,4,3,2]
    #   reshaped = reversed_part.reshape((3,2))
    #   Expected B == reshaped

    truncated = A[1:]
    reversed_part = truncated[::-1]
    reshaped = reversed_part.reshape((3, 2))

    assert B == reshaped



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
