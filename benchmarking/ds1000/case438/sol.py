import json
import math

from zinnia import *


@zk_circuit
def verify_solution(post: NDArray[float, 4], distance: NDArray[float, 4], result: float):
    # post = [2, 5, 6, 10]
    # distance = [50, 100, 500, 1000]
    # Pearson correlation coefficient:
    # r = cov(post, distance) / (std(post) * std(distance))

    n = 4
    mean_post = np.mean(post)
    mean_distance = np.mean(distance)

    # Compute covariance
    cov = 0.0
    for i in range(n):
        cov += (post[i] - mean_post) * (distance[i] - mean_distance)
    cov /= n

    # Compute standard deviations manually (np.std unavailable)
    var_post = 0.0
    var_distance = 0.0
    for i in range(n):
        var_post += (post[i] - mean_post) * (post[i] - mean_post)
        var_distance += (distance[i] - mean_distance) * (distance[i] - mean_distance)
    var_post /= n
    var_distance /= n

    std_post = math.sqrt(var_post)
    std_distance = math.sqrt(var_distance)

    pearson_r = cov / (std_post * std_distance)
    expected = pearson_r

    assert result == expected


if __name__ == '__main__':
    post = [2, 5, 6, 10]
    distance = [50, 100, 500, 1000]
    import numpy as np
    result = np.corrcoef(post, distance)[0][1]

    assert verify_solution(post, distance, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(post, distance, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
