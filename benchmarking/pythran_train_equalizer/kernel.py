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
@requires(lambda E, TrSyms, Niter, os, mu, wx, modes, adaptive, symbols, method: 0 <= Niter <= 1024)
@requires(lambda E, TrSyms, Niter, os, mu, wx, modes, adaptive, symbols, method: method != 0)
def train_equaliser(E: Complex[Array, 'N<=1024 M<=1024'], TrSyms: Integer, Niter: Integer, os: Integer, mu: Float, wx: Complex[Array, 'N<=1024 M<=1024 K<=1024'], modes: Integer[Array, 'N<=1024'], adaptive: Bool, symbols: Complex[Array, 'N<=1024 M<=1024'], method: str):
    if method == 'cma':
        errorfct = cma_error
    else:
        raise ValueError('Unknown method %s' % method)
    nmodes = E.shape[0]
    ntaps = wx.shape[-1]
    assert symbols.shape[0] == nmodes, 'symbols must be at least size of modes'
    assert wx.shape[0] == nmodes, 'wx needs to have at least as many dimensions as the maximum mode'
    assert E.shape[1] > TrSyms * os + ntaps, 'Field must be longer than the number of training symbols'
    assert modes.max() < nmodes, 'Maximum mode number must not be higher than number of modes'
    err = np.zeros((nmodes, TrSyms * Niter), dtype=E.dtype)
    for mode in modes:
        for it in range(Niter):
            for i in range(TrSyms):
                X = E[:, i * os:i * os + ntaps]
                Xest = apply_filter(X, wx[mode])
                err[mode, it * Niter + i] = errorfct(Xest, symbols[mode], i)
                wx[mode] += mu * np.conj(err[mode, it * Niter + i]) * X
                if adaptive and i > 0:
                    mu = adapt_step(mu, err[mode, it * Niter + i], err[mode, it * Niter + i - 1])
    return (err, wx, mu)

@zk_chip
def cma_error(Xest: Float, s1: Complex[Array, 'N<=1024'], i: Integer):
    d = s1[0].real - abs(Xest) ** 2
    return d * Xest

@zk_chip
def adapt_step(mu: Float, err_p: Float, err: Float):
    if err.real * err_p.real > 0 and err.imag * err_p.imag > 0:
        return mu
    else:
        return mu / (1 + mu * (err.real * err.real + err.imag * err.imag))

@zk_chip
def apply_filter(E: Complex[Array, 'N<=1024 M<=1024'], wx: Complex[Array, 'N<=1024 M<=1024']):
    pols = E.shape[0]
    Ntaps = wx.shape[1]
    Xest = E.dtype.type(0)
    for k in range(pols):
        for i in range(Ntaps):
            Xest += E[k, i] * np.conj(wx[k, i])
    return Xest
