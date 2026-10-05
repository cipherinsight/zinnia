# pythran program goodExpoMeasure.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy

@zk_circuit
@requires(lambda inRGB, sigma: sigma != 0)
def goodExpoMeasure(inRGB: Integer[Array, '20 20 20'], sigma: Float):
    """
    Compute the good exposition image quality measure on 1 input image.
    """
    R = inRGB[0, :, :].astype(numpy.float64)
    G = inRGB[1, :, :].astype(numpy.float64)
    B = inRGB[2, :, :].astype(numpy.float64)
    goodExpoR = numpy.exp(-(R - 128) ** 2 / sigma)
    goodExpoG = numpy.exp(-(G - 128) ** 2 / sigma)
    goodExpoB = numpy.exp(-(B - 128) ** 2 / sigma)
    goodExpo = goodExpoR * goodExpoG * goodExpoB
    goodExpo = (numpy.round(goodExpo, 2) * (2 ** 8 - 1)).astype(numpy.uint8)
    return goodExpo
