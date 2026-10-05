import json

from zinnia import *


@zk_circuit
def verify_solution(data: DynamicNDArray[float, 10, 1], result: DynamicNDArray[float, 3, 1]):
    bins = result.shape[0]
    bin_size = data.shape[0] // bins

    for b in range(bins):
        start = data.shape[0] - (b + 1) * bin_size
        total = 0.0
        for k in range(bin_size):
            total = total + data[start + k]
        assert result[b] == total / float(bin_size)


if __name__ == '__main__':
    data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    result = [5.0, 5.3333333333, 4.3333333333]

    assert verify_solution(data, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(data, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
