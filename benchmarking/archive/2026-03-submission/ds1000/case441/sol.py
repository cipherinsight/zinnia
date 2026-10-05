import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 4], number: int, is_contained: int):
    # a = [9, 2, 7, 0]
    # number = 0
    # Check if number exists in a
    found = 0
    for i in range(4):
        if a[i] == number:
            found = 1
    assert is_contained == found


if __name__ == '__main__':
    a = [9, 2, 7, 0]
    number = 0
    is_contained = 1

    assert verify_solution(a, number, is_contained)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, number, is_contained)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
