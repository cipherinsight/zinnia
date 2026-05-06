# Source: Pythran tests/cases/train_eq.py
# Original #pythran export: train_eq(complex64[][], int, int, float32, complex64[][], (complex64, complex64[]), bool)
# Migration notes: complex types and tuple-of-array param likely unsupported.
from zinnia import *

POLS = 1
LEN = 100
NTAPS = 10
NSYM = 4


@zk_circuit
def train_eq(E: NDArray[Float, 1, 100], TrSyms: int, os: int, mu: float,
             wx: NDArray[Float, 1, 10], errfctprs: tuple, adapt: bool):
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
    _zinnia_result = err, wx, mu
