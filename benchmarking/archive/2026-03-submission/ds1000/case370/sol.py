import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 3, 5], result: int):
    # a =
    # [
    #   [1, 2, 3, 4, 5],
    #   [1, 2, 3, 4, 5],
    #   [1, 2, 3, 4, 5]
    # ]
    #
    # Reference code:
    # result = np.isclose(a, a[0], atol=0).all()
    #
    # Meaning: check if all rows are identical.

    comparison = a == a[0]
    computed = np.all(comparison)

    assert (result == 1) == computed


if __name__ == '__main__':
    a = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
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
