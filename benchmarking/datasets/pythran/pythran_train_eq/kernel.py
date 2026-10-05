# pythran program train_eq.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def train_eq(E: Complex[Array, '1 16'], TrSyms: Integer, os: Integer, mu: Float, wx: Complex[Array, '1 10'], errfctprs: Tuple[Complex, Complex[Array, '4']], adapt: Boolean):
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
def train_eq2(E: Complex[Array, '1 16'], TrSyms: Integer, os: Integer, mu: Float, wx: Complex[Array, '1 10'], errfctprs: Tuple[Complex, Complex[Array, '4']], adapt: Boolean):
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

@zk_chip
def cma_error(Xest, R, symbs):
    return (R.real - abs(Xest) ** 2) * Xest

@zk_chip
def mcma_error(Xest, R, symbs):
    return symbs.dtype.type((R.real - Xest.real ** 2) * Xest.real + (R.imag - Xest.imag ** 2) * Xest.imag * 1j)
