# Source: Pythran tests/cases/train_equalizer.py
# Original #pythran export: train_equaliser(complex128[][], int, int, int, float64, complex128[][][], int[], bool, complex128[][], str)
# Migration notes: complex types and string method dispatch are likely unsupported by Zinnia.
from zinnia import *

NMODES = 2
LEN = 32
NTAPS = 8
NSYM = 4


def cma_error(Xest, s1, i):
    d = s1[0].real - abs(Xest) ** 2
    return d * Xest


def adapt_step(mu, err_p, err):
    if err.real * err_p.real > 0 and err.imag * err_p.imag > 0:
        return mu
    else:
        return mu / (1 + mu * (err.real * err.real + err.imag * err.imag))


def apply_filter(E, wx):
    pols = E.shape[0]
    Ntaps = wx.shape[1]
    Xest = E.dtype.type(0)
    for k in range(pols):
        for i in range(Ntaps):
            Xest += E[k, i] * np.conj(wx[k, i])
    return Xest


@zk_circuit
def train_equaliser(E: NDArray[Complex, 2, 32], TrSyms: int, Niter: int, os: int, mu: float,
                    wx: NDArray[Complex, 2, 2, 8], modes: NDArray[Integer, 2],
                    adaptive: bool, symbols: NDArray[Complex, 2, 4], method: str):
    if method == "cma":
        errorfct = cma_error
    else:
        raise ValueError("Unknown method %s" % method)
    nmodes = E.shape[0]
    ntaps = wx.shape[-1]
    assert symbols.shape[0] == nmodes, "symbols must be at least size of modes"
    assert wx.shape[0] == nmodes, "wx needs to have at least as many dimensions as the maximum mode"
    assert E.shape[1] > TrSyms * os + ntaps, "Field must be longer than the number of training symbols"
    assert modes.max() < nmodes, "Maximum mode number must not be higher than number of modes"
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
    _zinnia_result = err, wx, mu
