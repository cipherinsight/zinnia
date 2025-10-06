import json

from zinnia import *

@zk_circuit
def verify_solution(a: NDArray[int, 2, 3], result: NDArray[int, 6, 5]):
    # a = [[1, 0, 3],
    #      [2, 4, 1]]
    a_min = 0
    flat = [a[0,0], a[0,1], a[0,2], a[1,0], a[1,1], a[1,2]]  # flatten in C order

    for i in range(6):
        for j in range(5):
            expected = 1 if (flat[i] - a_min) == j else 0
            assert result[i, j] == expected



if __name__ == '__main__':
    a = [
        [1, 0, 3],
        [2, 4, 1]
    ]
    result = [
        [0, 1, 0, 0, 0],  # 1 → index 1
        [1, 0, 0, 0, 0],  # 0 → index 0
        [0, 0, 0, 1, 0],  # 3 → index 3
        [0, 0, 1, 0, 0],  # 2 → index 2
        [0, 0, 0, 0, 1],  # 4 → index 4
        [0, 1, 0, 0, 0]  # 1 → index 1
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
