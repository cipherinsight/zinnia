# Source: Pythran tests/cases/fbcorr_numpy.py
# Original #pythran export: fbcorr(float[][][][], float[][][][], float[][][][])
from zinnia import *

NI = 4
NR = 8
NC = 8
NCH = 4
NF = 2
H = 3
W = 3
OR = 6
OC = 6


@zk_circuit
def fbcorr(imgs: NDArray[Float, 4, 8, 8, 4],
           filters: NDArray[Float, 2, 3, 3, 4],
           output: NDArray[Float, 4, 2, 6, 6]):
    n_imgs, n_rows, n_cols, n_channels = imgs.shape
    n_filters, height, width, n_ch2 = filters.shape

    for ii in range(n_imgs):
        for rr in range(n_rows - height + 1):
            for cc in range(n_cols - width + 1):
                for hh in range(height):
                    for ww in range(width):
                        for jj in range(n_channels):
                            for ff in range(n_filters):
                                imgval = imgs[ii, rr + hh, cc + ww, jj]
                                filterval = filters[ff, hh, ww, jj]
                                output[ii, ff, rr, cc] += imgval * filterval
