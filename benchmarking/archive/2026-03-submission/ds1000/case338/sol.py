import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 25, 2], result: DynamicNDArray[int, 10, 2]):
    rows = result.shape[0] // 2
    cols = a.shape[0] // rows

    for r in range(rows):
        main_idx = r * cols + r
        anti_idx = r * cols + (rows - 1 - r)
        assert result[r] == a[main_idx]
        assert result[rows + r] == a[anti_idx]


if __name__ == '__main__':
    a = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]
    ]
    result = [
        [0, 6, 12, 18, 24],
        [4, 8, 12, 16, 20]
    ]
    assert verify_solution(a, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
