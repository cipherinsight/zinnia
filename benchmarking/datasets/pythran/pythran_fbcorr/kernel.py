# pythran program fbcorr_numpy.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def fbcorr(imgs: Float[Array, '8 8 8 8'], filters: Float[Array, '8 8 8 8'], output: Float[Array, '8 8 8 8']):
    n_imgs, n_rows, n_cols, n_channels = imgs.shape
    n_filters, height, width, n_ch2 = filters.shape
    'omp parallel for'
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
