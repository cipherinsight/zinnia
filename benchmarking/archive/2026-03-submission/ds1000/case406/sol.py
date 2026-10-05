import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 10], index: NDArray[int, 10], result: NDArray[int, 3]):
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # index = [0, 1, 0, 0, 0, 1, 1, 2, 2, 1]
    # Groups:
    #   i=0 → [1, 3, 4, 5] → max=5
    #   i=1 → [2, 6, 7, 10] → max=10
    #   i=2 → [8, 9] → max=9
    # Expected result = [5, 10, 9]

    # Precompute expected max for each group statically
    max0 = 0
    max1 = 0
    max2 = 0
    for i in range(10):
        if index[i] == 0 and a[i] > max0:
            max0 = a[i]
        if index[i] == 1 and a[i] > max1:
            max1 = a[i]
        if index[i] == 2 and a[i] > max2:
            max2 = a[i]
    expected = [max0, max1, max2]
    assert result == expected


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    index = [0, 1, 0, 0, 0, 1, 1, 2, 2, 1]
    result = [5, 10, 9]

    assert verify_solution(a, index, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, index, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
