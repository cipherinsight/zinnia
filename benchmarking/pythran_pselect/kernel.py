from zinnia import *

@zk_circuit
@requires(lambda n: 0 <= n <= 1024)
def pselect(n: Integer):
    l = list()
    for k in (n, not n):
        if k:
            a = sel0
        else:
            a = sel1
        a(l)
    return l

@zk_chip
def sel0(n: List[Float]):
    n.append(1)

@zk_chip
def sel1(n: List[Float]):
    n.append(2.0)
