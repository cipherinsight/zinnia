# Source: Pythran tests/cases/ldpc_decoder.py
# Original #pythran export: Decoding_logBP(int[:,:], int list list, int list list, float[:,:], float[:], int)
# Migration notes: chose the largest exported function (Decoding_logBP); helpers included.
from zinnia import *
from math import log, tanh

M = 8
N = 16


def phi0(x):
    x = abs(x)
    if (x < 9.08e-5):
        return 10
    else:
        return -log(tanh(x / 2))


def G(Lq):
    X = sum(phi0(e) for e in Lq)
    s = np.prod(np.sign(Lq))
    return s * phi0(X)


def BinaryProduct(X, Y):
    A = X.dot(Y)
    return A % 2


def InCode(H, x):
    return (BinaryProduct(H, x) == 0).all()


@zk_circuit
def Decoding_logBP(H: NDArray[Integer, 8, 16], Bits: list, Nodes: list,
                   Lq: NDArray[Float, 8, 16], Lc: NDArray[Float, 16], max_iter: int = 1):
    m, n = H.shape

    if not len(Lc) == n:
        raise ValueError('La taille de y doit correspondre au nombre de colonnes de H')

    if m >= n:
        raise ValueError('H doit avoir plus de colonnes que de lignes')

    Lr = np.zeros(shape=(m, n))
    count = 0
    Lq += Lc

    while True:
        count += 1

        for i in range(m):
            Ni = Bits[i]
            for j in Ni:
                Nij = list(Ni)
                Nij.remove(j)
                Lr[i, j] = G(Lq[i][Nij])
        Lr = np.clip(Lr, -100, 100)

        for j in range(n):
            Mj = Nodes[j]

            for i in Mj:
                Mji = list(Mj)
                Mji.remove(i)

                Lq[i, j] = Lc[j] + sum(Lr[Mji][:, j])

        extrinsic = np.empty(n)
        for j in range(n):
            Mj = Nodes[j]

            extrinsic[j] = sum(Lr[Mj][:, j])

        L_posteriori = extrinsic + Lc
        x = np.array(extrinsic <= 0).astype(int)
        product = InCode(H, x)

        if product or count >= max_iter:
            break
    _zinnia_result = np.array(L_posteriori <= 0).astype(int), Lq - Lc, extrinsic, product
