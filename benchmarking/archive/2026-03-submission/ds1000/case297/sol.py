import json

from zinnia import *


@zk_circuit
def verify_solution(a: DynamicNDArray[int, 3, 1], result: DynamicNDArray[int, 15, 2]):
    shifted = a - a.min()
    columns = np.stack((
        result[0:15:5],
        result[1:15:5],
        result[2:15:5],
        result[3:15:5],
        result[4:15:5],
    ), axis=0)
    for value in range(5):
        assert (columns[value] == 1) == (shifted == value)



if __name__ == '__main__':
    a = [1, 2, 5]
    result = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ]
    assert verify_solution(a, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
