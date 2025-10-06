import json

from zinnia import *


@zk_circuit
def verify_solution(x: NDArray[float, 13], result: NDArray[float, 10]):
    # x = [-2, -1.4, -1.1, 0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2]
    # Remove negative elements → keep x >= 0
    # Expected:
    # [0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2]

    filtered = []
    for i in range(x.shape[0]):
        if x[i] >= 0:
            filtered.append(x[i])

    expected = filtered
    assert result == expected


if __name__ == '__main__':
    x = [-2, -1.4, -1.1, 0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2]
    result = [0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2]

    assert verify_solution(x, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(x, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
