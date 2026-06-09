from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.
import numpy as np

@zk_circuit
def train_eq(E: Complex[Array, 'N<=1024 M<=1024'], TrSyms: Complex[Array, 'N<=1024'], os: Float, mu: Float, wx: Complex[Array, 'N<=1024'], errfctprs: Float, adapt: Float):
    Ntaps = wx.shape[1]
    pols = wx.shape[0]
    R, symbs = errfctprs
    err = np.zeros(TrSyms, dtype=E.dtype)
    for i in range(TrSyms):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(np.conj(wx) * X)
        err[i] = (R.real - abs(Xest) ** 2) * Xest
        wx += mu * np.conj(err[i]) * X
        if adapt and i > 0:
            if err[i].real * err[i - 1].real > 0 and err[i].imag * err[i - 1].imag > 0:
                pass
            else:
                mu = np.float32(mu / (1 + mu * (err[i].real * err[i].real + err[i].imag * err[i].imag)))
    return (err, wx, mu)

@zk_circuit
def train_eq2(E: Complex[Array, 'N<=1024 M<=1024'], TrSyms: Complex[Array, 'N<=1024'], os: Float, mu: Float, wx: Complex[Array, 'N<=1024'], errfctprs: Float, adapt: Float):
    Ntaps = wx.shape[1]
    pols = wx.shape[0]
    R, symbs = errfctprs
    err = np.zeros(TrSyms, dtype=E.dtype)
    for i in range(TrSyms):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(np.conj(wx) * X)
        err[i] = (R.real - abs(Xest) ** 2) * Xest
        wx += mu * np.conj(err[i]) * X
        if adapt and i > 0:
            if err[i].real * err[i - 1].real > 0 and err[i].imag * err[i - 1].imag > 0:
                pass
            else:
                mu = np.float32(mu / (1 + mu * (err[i].real * err[i].real + err[i].imag * err[i].imag)))
    return (err, wx, mu)

@zk_circuit
def cma_error(Xest: Float, R: Integer, symbs: Float):
    return (R.real - abs(Xest) ** 2) * Xest

@zk_circuit
def mcma_error(Xest: Float, R: Integer, symbs: Complex[Array, 'N<=1024']):
    return symbs.dtype.type((R.real - Xest.real ** 2) * Xest.real + (R.imag - Xest.imag ** 2) * Xest.imag * 1j)
