# pythran program vibr_energy.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy

@zk_circuit
def calculate_vibr_energy(harmonic: Float[Array, '1000000'], anharmonic: Float[Array, '1000000'], i: Float[Array, '1000000']):
    return numpy.exp(-harmonic * i - anharmonic * i ** 2)
