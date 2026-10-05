from zinnia import *

@zk_chip
def hits(x1: Integer, y1: Integer, x2: Integer, y2: Integer):
    """Check whether a queen positioned at (x1, y1) will hit a queen at position (x2, y2)"""
    return x1 == x2 or y1 == y2 or abs(x1 - x2) == abs(y1 - y2)

@zk_chip
def hitsany(x: Integer, y: Integer, queens_x: List[Integer], queens_y: List[Integer]):
    """Check whether a queen positioned at (x1, y1) will hit any other queen"""
    for i in range(len(queens_x)):
        if hits(x, y, queens_x[i], queens_y[i]):
            return True
    return False

@zk_chip
def _solve(n: Integer, queens_x: List[Integer], queens_y: List[Integer]):
    """Solve the queens puzzle"""
    if n == 0:
        return True
    for x in range(1, 9):
        for y in range(1, 9):
            if not hitsany(x, y, queens_x, queens_y):
                queens_x.append(x)
                queens_y.append(y)
                if _solve(n - 1, queens_x, queens_y):
                    return True
                queens_x.pop()
                queens_y.pop()
    return False

@zk_circuit
@requires(lambda n: 0 <= n <= 1024)
def solve(n: Integer):
    queens_x = []
    queens_y = []
    if _solve(n, queens_x, queens_y):
        return (queens_x, queens_y)
    else:
        return None
