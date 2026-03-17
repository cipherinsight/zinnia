import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 15, 2], result: int):
    col0 = a[0:15:3]
    col1 = a[1:15:3]
    col2 = a[2:15:3]
    assert result == ((col0 == col1).all() and (col0 == col2).all())


if __name__ == '__main__':
    a = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
    ]
    result = True
    assert verify_solution(a, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
