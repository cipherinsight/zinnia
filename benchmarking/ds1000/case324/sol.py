import json
import math

from zinnia import *


@zk_circuit
def verify_solution(degree: float, result: float):
    # degree = 90
    # Reference formula: result = np.cos(np.deg2rad(degree))
    # np.deg2rad(x) = x * π / 180
    pi = 3.141592653589793
    rad = degree * pi / 180.0
    computed = math.cos(rad)

    assert result == computed


if __name__ == '__main__':
    degree = 90
    result = 0.0
    assert verify_solution(degree, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(degree, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
