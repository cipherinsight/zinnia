# Source: Pythran tests/cases/lapl2d.py
# Original #pythran export: lapl2d(float[,], float[,], int)
from zinnia import *

N = 16


@zk_circuit
def lapl2d(In: NDArray[Float, 16, 16], Out: NDArray[Float, 16, 16], niter: int):
    siz = In.shape[0]
    h2 = (1. / siz) ** 2
    for it in range(0, niter):
        Out[1:siz - 1, 1:siz - 1] = h2 * (
            In[0:siz - 2, 1:siz - 1] + In[1:siz - 1, 0:siz - 2] -
            4.0 * In[1:siz - 1, 1:siz - 1] +
            In[2:siz, 1:siz - 1] + In[1:siz - 1, 2:siz])
        In, Out = Out, In
