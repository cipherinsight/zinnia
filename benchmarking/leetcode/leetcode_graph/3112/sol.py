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
from zinnia import *


@zk_circuit
def verify_solution(
        graph: Public[NDArray[int, 10, 10]],
        disappear: Public[NDArray[int, 10]],
        answers: Public[NDArray[int, 10]]
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
