from zinnia import zk_circuit

@zk_circuit
def  Prime(n:int, y:int):
    assert n>=1 and n<=10000
    assert y==0 or y==1
    if n==1:
        assert y==0
    elif n==2:
        assert y==1
    else:
        cnt=0
        for i in range(2,101):
            if n%i==0:
                cnt=1
            if i*i>n:
                break
        if cnt==0:
            assert y == 1
        else:
            assert y == 0
