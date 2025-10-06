import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 2, 2], result: NDArray[int, 2, 2]):
    # a =
    # [[1, 0],
    #  [0, 2]]
    # Expected result =
    # [[0, 1],
    #  [1, 0]]

    # Step 1: Compute the minimum value
    min_val = a.min()

    # Step 2: Collect indices where a[i, j] == min_val
    # Since argwhere is unavailable, we manually build result
    expected = np.zeros((2, 2), dtype=int)
    idx = 0
    for i in range(2):
        for j in range(2):
            if a[i, j] == min_val:
                expected[idx, 0] = i
                expected[idx, 1] = j
                idx += 1

    # Step 3: Verify the result
    assert result == expected


if __name__ == '__main__':
    a = [
        [1, 0],
        [0, 2]
    ]
    result = [
        [0, 1],
        [1, 0]
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
