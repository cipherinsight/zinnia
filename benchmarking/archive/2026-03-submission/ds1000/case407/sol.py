import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 10], accmap: NDArray[int, 10], result: NDArray[int, 3]):
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(10):
        sum0 += a[i] * (accmap[i] == 0)
        sum1 += a[i] * (accmap[i] == 1)
        sum2 += a[i] * (accmap[i] == 2)

    expected = [sum0, sum1, sum2]
    assert result == expected


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    accmap = [0, 1, 0, 0, 0, 6, 8, 2, 2, 1]
    result = [13, 12, 17]

    assert verify_solution(a, accmap, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, accmap, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
