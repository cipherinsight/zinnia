import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 2, 2], power: int, result: NDArray[int, 2, 2]):
    # a = [[0, 1],
    #      [2, 3]]
    # power = 5
    # Reference formula: a = a ** power

    computed = a ** power

    assert result == computed


if __name__ == '__main__':
    a = [
        [0, 1],
        [2, 3]
    ]
    power = 5
    result = [
        [0, 1],
        [32, 243]
    ]
    assert verify_solution(a, power, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, power, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
