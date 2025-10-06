import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 3], b: NDArray[int, 3], c: NDArray[int, 3], result: NDArray[int, 3]):
    # a = [10, 20, 30]
    # b = [30, 20, 20]
    # c = [50, 20, 40]
    # Reference formula: result = np.max([a, b, c], axis=0)
    stacked = np.array([a.tolist(), b.tolist(), c.tolist()])
    computed = np.max(stacked, axis=0)

    assert result == computed


if __name__ == '__main__':
    a = [10, 20, 30]
    b = [30, 20, 20]
    c = [50, 20, 40]
    result = [50, 20, 40]
    assert verify_solution(a, b, c, result)

    # Parse inputs for compilation
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, b, c, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
