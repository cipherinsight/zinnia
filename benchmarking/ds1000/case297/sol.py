import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 3], result: NDArray[int, 3, 5]):
    # a = [1, 2, 5]
    a_min = a.min()
    # temp = a - a_min = [0, 1, 4]
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            expected = 1 if (a[i] - a_min) == j else 0
            assert result[i, j] == expected



if __name__ == '__main__':
    a = [1, 2, 5]
    result = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ]
    assert verify_solution(a, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
