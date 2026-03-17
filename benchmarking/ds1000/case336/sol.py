import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 25, 2], result: DynamicNDArray[int, 5, 1]):
    n = result.shape[0]
    side = a.shape[0] // n
    for r in range(n):
        idx = r * side + (side - 1 - r)
        assert result[r] == a[idx]


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
