# pythran program ldpc_decoder.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np
from math import log, tanh

@zk_chip
def phi0(x):
    x = abs(x)
    if x < 9.08e-05:
        return 10
    else:
        return -log(tanh(x / 2))

@zk_chip
def G(Lq):
    X = sum((phi0(e) for e in Lq))
    s = np.prod(np.sign(Lq))
    return s * phi0(X)

@zk_chip
def BinaryProduct(X, Y):
    """ Binary Matrices or Matrix-vector product in Z/2Z. Works with scipy.sparse.csr_matrix matrices X,Y too."""
    A = X.dot(Y)
    return A % 2

@zk_chip
def InCode(H, x):
    """ Computes Binary Product of H and x. If product is null, x is in the code.
        Returns appartenance boolean. 
    """
    return (BinaryProduct(H, x) == 0).all()

@zk_circuit
@requires(lambda H, Bits, Nodes, Lq, Lc, max_iter: 0 <= max_iter <= 1024)
def Decoding_logBP(H: Integer[Array, '8 8'], Bits: Integer[Array, '8 8'], Nodes: Integer[Array, '8 8'], Lq: Float[Array, '8 8'], Lc: Float[Array, '8'], max_iter: Integer=1):
    """ Decoding function using Belief Propagation algorithm (logarithmic version)
    IMPORTANT: if H is large (n>1000), H should be scipy.sparse.csr_matrix object to speed up calculations
    (highly recommanded. )
    -----------------------------------
    Parameters:
    
    H: 2D-array (OR scipy.sparse.csr_matrix object) Parity check matrix, shape = (m,n) 
    y: n-vector recieved after transmission in the channel. (In general, returned 
    by Coding Function)
    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN used in coding.
    
    max_iter: (default = 1) max iterations of the main loop. Increase if decoding is not error-free.
     """
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
    return (np.array(L_posteriori <= 0).astype(int), Lq - Lc, extrinsic, product)
