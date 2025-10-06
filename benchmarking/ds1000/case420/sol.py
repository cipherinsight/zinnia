import json

from zinnia import *


@zk_circuit
def verify_solution(x: float, result: float):
    x_min = 0.0
    x_max = 1.0
    expected = x_min
    if x > x_max:
        expected = x_max
    elif x >= x_min:
        expected = 3 * x ** 2 - 2 * x ** 3
    assert result == expected


if __name__ == '__main__':
    x = 0.25
    result = 3 * (x ** 2) - 2 * (x ** 3)

    assert verify_solution(x, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(x, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
