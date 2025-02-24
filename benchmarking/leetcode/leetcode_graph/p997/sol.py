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
from zinnia import *

@zk_circuit
def verify_solution(
        trust_graph: Public[NDArray[int, 10, 10]],
        judge_id: int
):
    for i in range(10):
        if i == judge_id - 1:
            continue
        if trust_graph[i][judge_id - 1] == 0:
            assert False
        if trust_graph[judge_id - 1][i] == 1:
            assert False
