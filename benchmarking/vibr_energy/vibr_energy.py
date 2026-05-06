# Source: Pythran tests/cases/vibr_energy.py
# Original #pythran export: calculate_vibr_energy(float[], float[], int[])
from zinnia import *

N = 64


@zk_circuit
def calculate_vibr_energy(harmonic: NDArray[Float, 64], anharmonic: NDArray[Float, 64], i: NDArray[Integer, 64]):
    _zinnia_result = np.exp(-harmonic * i - anharmonic * (i ** 2))
