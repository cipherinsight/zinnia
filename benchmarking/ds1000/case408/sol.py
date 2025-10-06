import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 10], index: NDArray[int, 10], result: NDArray[int, 3]):
    n = a.shape[0]
    # First pass: seed minima from the first occurrence of each group (no infinities needed)
    found0 = False
    found1 = False
    found2 = False
    min0 = 0
    min1 = 0
    min2 = 0
    for i in range(n):
        if (index[i] == 0) and (not found0):
            min0 = a[i]
            found0 = True
        if (index[i] == 1) and (not found1):
            min1 = a[i]
            found1 = True
        if (index[i] == 2) and (not found2):
            min2 = a[i]
            found2 = True

    # Ensure each group exists (true for this instantiated case)
    assert found0 and found1 and found2

    # Second pass: refine minima
    for i in range(n):
        if index[i] == 0 and a[i] < min0:
            min0 = a[i]
        if index[i] == 1 and a[i] < min1:
            min1 = a[i]
        if index[i] == 2 and a[i] < min2:
            min2 = a[i]

    expected = [min0, min1, min2]
    assert result == expected


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
