# Source: Pythran tests/cases/fbcorr.py
# Original #pythran export: fbcorr(float list list list list, float list list list list)
from zinnia import *


@zk_circuit
def fbcorr(imgs: NDArray[Float, 4, 4, 4, 4], filters: NDArray[Float, 4, 4, 4, 4]):
    n_imgs, n_rows, n_cols, n_channels = (len(imgs), len(imgs[0]), len(imgs[0][0]), len(imgs[0][0][0]))
    n_filters, height, width, n_ch2 = (len(filters), len(filters[0]), len(filters[0][0]), len(filters[0][0][0]))
    output = [[[[0 for i in range(n_cols - width + 1)] for j in range(n_rows - height + 1)] for k in range(n_filters)] for l in range(n_imgs)]
    for ii in range(n_imgs):
        for rr in range(n_rows - height + 1):
            for cc in range(n_cols - width + 1):
                for hh in range(height):
                    for ww in range(width):
                        for jj in range(n_channels):
                            for ff in range(n_filters):
                                imgval = imgs[ii][rr + hh][cc + ww][jj]
                                filterval = filters[ff][hh][ww][jj]
                                output[ii][ff][rr][cc] += imgval * filterval
    _zinnia_result = output
