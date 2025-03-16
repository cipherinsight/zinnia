from zinnia import zk_circuit,NDArray


@zk_circuit
def Graph(A:NDArray[int, 10, 10],a:int,b:int, y:int):
    assert a>=0 and a<=9
    assert b>=0 and b<=9
    assert y==0 or y==1
    assert A[a,b]==y