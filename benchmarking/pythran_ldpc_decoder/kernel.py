from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, denominator-obligation, domain-obligation, in-place-mutation, loop-bounded, matmul-dot
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Declare shared matrix/vector dimensions explicitly; when not expressible as @requires, encode with matching symbolic annotation names and keep as review obligation.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.
import numpy as np
from math import log, tanh

@zk_chip
def phi0(x: Float):
    x = abs(x)
    if x < 9.08e-05:
        return 10
    else:
        return -log(tanh(x / 2))

@zk_chip
def G(Lq: Float[Array, 'N<=1024']):
    X = sum((phi0(e) for e in Lq))
    s = np.prod(np.sign(Lq))
    return s * phi0(X)

@zk_chip
def BinaryProduct(X: Integer[Array, 'N<=1024 M<=1024'], Y: Integer[Array, 'N<=1024']):
    """ Binary Matrices or Matrix-vector product in Z/2Z. Works with scipy.sparse.csr_matrix matrices X,Y too."""
    A = X.dot(Y)
    return A % 2

@zk_chip
def InCode(H: Integer[Array, 'N<=1024 M<=1024'], x: Integer[Array, 'N<=1024']):
    """ Computes Binary Product of H and x. If product is null, x is in the code.
        Returns appartenance boolean. 
    """
    return (BinaryProduct(H, x) == 0).all()

@zk_circuit
@requires(lambda H, Bits, Nodes, Lq, Lc, max_iter: 0 <= max_iter <= 1024)
def Decoding_logBP(H: Integer[Array, 'N<=1024 M<=1024'], Bits: Integer[Array, 'N<=1024 M<=1024'], Nodes: Integer[Array, 'N<=1024 M<=1024'], Lq: Float[Array, 'N<=1024 M<=1024'], Lc: Float[Array, 'N<=1024'], max_iter: Integer=1):
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
