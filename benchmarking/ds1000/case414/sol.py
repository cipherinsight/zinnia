import json

from zinnia import *


@zk_circuit
def verify_solution(data: DynamicNDArray[float, 10, 1], result: DynamicNDArray[float, 3, 1]):
    expected = data[:9].reshape((3, 3)).sum(axis=1) / 3
    assert result == expected


if __name__ == '__main__':
    data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    result = [3.6666666666666667, 6, 4]

    assert verify_solution(data, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(data, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
