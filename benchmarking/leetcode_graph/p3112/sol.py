# 3112. Minimum Time to Visit Disappearing Nodes
# Medium
# Topics
# Companies
# Hint
#
# There is an undirected graph of n nodes. You are given a 2D array edges, where edges[i] = [ui, vi, lengthi] describes an edge between node ui and node vi with a traversal time of lengthi units.
#
# Additionally, you are given an array disappear, where disappear[i] denotes the time when the node i disappears from the graph and you won't be able to visit it.
#
# Note that the graph might be disconnected and might contain multiple edges.
#
# Return the array answer, with answer[i] denoting the minimum units of time required to reach node i from node 0. If node i is unreachable from node 0 then answer[i] is -1.
import json

from zinnia import *


@zk_circuit
def verify_solution(
        graph: NDArray[int, 10, 10],
        disappear: NDArray[int, 10],
        answers: NDArray[int, 10]
):
    n = 10
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if graph[i][k] != -1 and graph[k][j] != -1:
                    graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
    for i in range(n):
        if graph[0][i] != -1:
            assert answers[i] == -1
        elif disappear[i] <= graph[0][i]:
            assert answers[i] == graph[0][i]
        else:
            assert answers[i] == -1


# entries = ZKCircuit.from_method(verify_solution).argparse(
#     np.array([
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     ]),
#     np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#     np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
# ).entries
#
# json_dict = {}
# for entry in entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
#
# json_dict = {}
# json_dict["graph"] = [int(x) for x in np.array([
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     ]).flatten().tolist()]
# json_dict["disappear"] = [int(x) for x in np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).tolist()]
# json_dict["answers"] = [int(x) for x in np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]).tolist()]
# print(json.dumps(json_dict, indent=2))

