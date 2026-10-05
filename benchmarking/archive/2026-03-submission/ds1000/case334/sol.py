import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 3, 1], b: DynamicNDArray[int, 3, 1], c: DynamicNDArray[int, 3, 1], result: DynamicNDArray[float, 3, 1]):
    expected = np.stack((a, b, c), axis=0).sum(axis=0) / 3
    assert result == expected


if __name__ == '__main__':
    a = [10, 20, 30]
    b = [30, 20, 20]
    c = [50, 20, 40]
    result = [30.0, 20.0, 30.0]
    assert verify_solution(a, b, c, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, b, c, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
