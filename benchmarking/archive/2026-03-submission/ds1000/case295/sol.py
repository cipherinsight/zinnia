import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 3], result: NDArray[int, 3, 4]):
    for i in range(3):
        for j in range(4):
            expected = 1 if a[i] == j else 0
            assert result[i][j] == expected



if __name__ == '__main__':
    a = [1, 0, 3]
    result = [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
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
