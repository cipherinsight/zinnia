# In a town, there are n people labeled from 1 to n. There is a rumor that one of these people is secretly the town judge.
#
# If the town judge exists, then:
#
#     The town judge trusts nobody.
#     Everybody (except for the town judge) trusts the town judge.
#     There is exactly one person that satisfies properties 1 and 2.
#
# You are given an array trust where trust[i] = [ai, bi] representing that the person labeled ai trusts the person labeled bi. If a trust relationship does not exist in trust array, then such a trust relationship does not exist.
#
# Return the label of the town judge if the town judge exists and can be identified, or return -1 otherwise.
import json

from zinnia import *

@zk_circuit
def verify_solution(
        trust_graph: NDArray[int, 10, 10],
        judge_id: int
):
    for i in range(10):
        for j in range(10):
            if i == judge_id - 1 and i != j:
                assert trust_graph[i][j] == 0
            if j == judge_id - 1 and i != j:
                assert trust_graph[i][j] == 1


# assert verify_solution(
#     np.array([
#         [1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
#         [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
#         [0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
#         [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
#         [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
#         [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
#         [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     ]),
#     10
# )
# entries = ZKCircuit.from_method(verify_solution).argparse(
#     np.array([
#         [1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
#         [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
#         [0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
#         [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
#         [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
#         [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
#         [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     ]),
#     10
# ).entries
#
# json_dict = {}
# for entry in entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
#
# json_dict = {}
# json_dict["graph"] = [int(x) for x in np.array([
#         [1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
#         [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
#         [0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
#         [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
#         [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
#         [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
#         [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     ]).flatten().tolist()]
# json_dict["judge_id"] = 10
# print(json.dumps(json_dict, indent=2))

