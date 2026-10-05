from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
import numpy

@zk_circuit
@requires(lambda inRGB, sigma: sigma != 0)
def goodExpoMeasure(inRGB: Float[Array, 'C<=3 H<=1024 W<=1024'], sigma: Float):
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
