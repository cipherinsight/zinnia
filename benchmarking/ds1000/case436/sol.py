import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 6, 2], mask: DynamicNDArray[int, 6, 2]):
    a0 = a[0:6:2]
    a1 = a[1:6:2]
    row_max = np.stack((a0, a1), axis=0).max(axis=0)
    m0 = mask[0:6:2]
    m1 = mask[1:6:2]
    assert (m0 == 1) == (a0 == row_max)
    assert (m1 == 1) == (a1 == row_max)


if __name__ == '__main__':
    a = [
        [0, 1],
        [2, 1],
        [4, 8]
    ]
    mask = [
        [0, 1],
        [1, 0],
        [0, 1]
    ]

    assert verify_solution(a, mask)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, mask)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
