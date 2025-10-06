import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 2, 3], result: int):
    # a = [[10, 50, 30],
    #      [60, 20, 40]]
    # Flattened array (C order): [10, 50, 30, 60, 20, 40]
    # Maximum value = 60 → raveled index = 3

    computed = a.argmax()
    assert result == computed



if __name__ == '__main__':
    a = [
        [10, 50, 30],
        [60, 20, 40]
    ]
    result = 3
    assert verify_solution(a, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
