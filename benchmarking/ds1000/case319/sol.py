import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 3, 2, 2], permutation: NDArray[int, 3], result: NDArray[int, 3, 2, 2]):
    # We want: result[k, r, s] == a[c[k], r, s], where c[k] is the inverse permutation of `permutation`.
    # Compute c[k] = sum_i i * [permutation[i] == k]
    # Then select a[c[k], r, s] by indicator sum over t in 0..3.

    for k in range(3):
        # Build inverse index c[k] via one-hot indicator
        ck = 0
        for i in range(3):
            is_target = 1 if permutation[i] == k else 0
            ck = ck + i * is_target

        # For each inner position (r, s), select a[ck, r, s]
        for r in range(2):
            for s in range(2):
                selected = 0
                for t in range(3):
                    ind = 1 if ck == t else 0
                    selected = selected + a[t, r, s] * ind
                assert result[k, r, s] == selected


if __name__ == '__main__':
    a = [
        [[10, 20],
         [30, 40]],
        [[6, 7],
         [8, 9]],
        [[10, 11],
         [12, 13]]
    ]
    permutation = [1, 0, 2]
    result = [
        [[6, 7],
         [8, 9]],
        [[10, 20],
         [30, 40]],
        [[10, 11],
         [12, 13]]
    ]
    assert verify_solution(a, permutation, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, permutation, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
