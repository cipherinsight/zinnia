import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[float, 3], result: NDArray[int, 3, 3]):
    # a = [1.5, -0.4, 1.3]
    # sorted unique values: [-0.4, 1.3, 1.5]
    vals = [-0.4, 1.3, 1.5]

    for i in range(3):
        for j in range(3):
            expected = 1 if a[i] == vals[j] else 0
            assert result[i, j] == expected



if __name__ == '__main__':
    a = [1.5, -0.4, 1.3]
    result = [
        [0, 0, 1],  # 1.5 corresponds to rightmost column
        [1, 0, 0],  # -0.4 corresponds to leftmost column
        [0, 1, 0]  # 1.3 corresponds to middle column
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
