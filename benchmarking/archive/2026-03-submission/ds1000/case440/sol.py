import json
import math

from zinnia import *


@zk_circuit
def verify_solution(Y: DynamicNDArray[int, 36, 3], X: DynamicNDArray[float, 12, 2]):
    diag_flat = np.concatenate((Y[0:9:4], Y[9:18:4], Y[18:27:4], Y[27:36:4]), axis=0)
    expected_order = np.concatenate((X[0:12:4], X[1:12:4], X[2:12:4], X[3:12:4]), axis=0)
    assert diag_flat == (expected_order ** 2)


if __name__ == '__main__':
    Y = [
        [[81, 63, 63],
         [63, 49, 49],
         [63, 49, 49]],

        [[4, 12, 8],
         [12, 36, 24],
         [8, 24, 16]],

        [[25, 35, 25],
         [35, 49, 35],
         [25, 35, 25]],

        [[25, 30, 10],
         [30, 36, 12],
         [10, 12, 4]],
    ]
    X = [
        [9.0, 2.0, 5.0, 5.0],
        [7.0, 6.0, 7.0, 6.0],
        [7.0, 4.0, 5.0, 2.0],
    ]

    assert verify_solution(Y, X)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(Y, X)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
