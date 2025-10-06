import json

from zinnia import *

@zk_circuit
def verify_solution(a: NDArray[int, 2, 5], permutation: NDArray[int, 5], result: NDArray[int, 2, 5]):
    # Goal: permuted[:, j] = a[:, c[j]], where c is the inverse permutation of `permutation`.
    # Construct c[j] via one-hot indicators:
    #   c[j] = sum_i i * [permutation[i] == j]
    # Then select a[:, c[j]] using another indicator sum over t in 0..5.

    for j in range(5):
        # Build c[j] as an integer via equality indicators
        cj = 0
        for i in range(5):
            is_target = 1 if permutation[i] == j else 0
            cj = cj + i * is_target

        # Row 0: select a[0, cj] by summing over all t with indicator [cj == t]
        sel_val_r0 = 0
        for t in range(5):
            ind = 1 if cj == t else 0
            sel_val_r0 = sel_val_r0 + a[0, t] * ind
        assert result[0, j] == sel_val_r0

        # Row 1: same selection for row 1
        sel_val_r1 = 0
        for t in range(5):
            ind = 1 if cj == t else 0
            sel_val_r1 = sel_val_r1 + a[1, t] * ind
        assert result[1, j] == sel_val_r1


if __name__ == '__main__':
    a = [
        [10, 20, 30, 40, 50],
        [6, 7, 8, 9, 10]
    ]
    permutation = [0, 4, 1, 3, 2]
    result = [
        [10, 30, 50, 40, 20],
        [6, 8, 10, 9, 7]
    ]
    assert verify_solution(a, permutation, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, permutation, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
