import json
import math

from zinnia import *

@zk_circuit
def verify_solution(area: int, expected_l: int, expected_w: int):
    w = area
    for i in range(1, 401):
        if area % i == 0:
            w = i
        if i * i >= area:
            break
    answer_l = area // w
    answer_w = w
    if answer_w > answer_l:
        answer_l, answer_w = answer_w, answer_l
    assert answer_l == expected_l and answer_w == expected_w


# verify_solution(351, 27, 13)
# entries = ZKCircuit.from_method(verify_solution).argparse(351, 27, 13).entries
# json_dict = {}
# for entry in entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
