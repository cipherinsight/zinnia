from zinnia import zk_circuit


@zk_circuit
def DP(n:int, y:int):
    assert n>=1 and n<=50
    if n==1:
        assert y==1
    elif n==2:
        assert y == 2
    else:
        a=1
        b=2
        for i in range(3,50):
            temp=b
            b=a+b
            a=temp
            if i==n:
                assert b==y
