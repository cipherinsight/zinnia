import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 6, 2], mask: DynamicNDArray[int, 6, 2]):
    rows = a.shape[0] // 2
    for r in range(rows):
        left_idx = 2 * r
        right_idx = left_idx + 1

        left = a[left_idx]
        right = a[right_idx]

        row_min = left if left <= right else right

        assert (mask[left_idx] == 1) == (left == row_min)
        assert (mask[right_idx] == 1) == (right == row_min)


if __name__ == '__main__':
    a = [
        [0, 1],
        [2, 1],
        [4, 8]
    ]
    mask = [
        [1, 0],
        [0, 1],
        [1, 0]
    ]

    assert verify_solution(a, mask)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, mask)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
