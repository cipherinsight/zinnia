# A web developer needs to know how to design a web page's size. So, given a specific rectangular web page’s area, your job by now is to design a rectangular web page, whose length L and width W satisfy the following requirements:
#
#     The area of the rectangular web page you designed must equal to the given target area.
#     The width W should not be larger than the length L, which means L >= W.
#     The difference between length L and width W should be as small as possible.
#
# Return an array [L, W] where L and W are the length and width of the web page you designed in sequence.
# Suppose that area <= 1000000
import json
import math

from zinnia import *

@zk_circuit
def verify_solution(area: int, expected_l: int, expected_w: int):
    w = area
    for i in range(1, 1001):
        if area % i == 0:
            w = i
        if i * i >= area:
            break
    answer_l = area // w
    answer_w = w
    if answer_w > answer_l:
        answer_l, answer_w = answer_w, answer_l
    assert answer_l == expected_l and answer_w == expected_w


# verify_solution(999, 37, 27)
# entries = ZKCircuit.from_method(verify_solution).argparse(999, 37, 27).entries
# json_dict = {}
# for entry in entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
