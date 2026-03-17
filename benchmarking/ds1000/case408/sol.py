import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 10, 1], index: DynamicNDArray[int, 10, 1], result: DynamicNDArray[int, 3, 1]):
    n = a.shape[0]
    groups = result.shape[0]
    for g in range(groups):
        found = False
        min_v = 0
        for i in range(n):
            idx = index[i]
            if idx < 0:
                idx = idx + groups
            if idx == g:
                if not found:
                    min_v = a[i]
                    found = True
                elif a[i] < min_v:
                    min_v = a[i]
        assert found
        assert result[g] == min_v


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    index = [0, 1, 0, 0, 0, 3, 4, 2, 2, 1]
    result = [1, 2, 8]

    assert verify_solution(a, index, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, index, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
