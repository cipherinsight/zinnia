# pythran program lapl2d.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda In, Out, niter: 0 <= niter <= 1024)
def lapl2d(In: Float[Array, '4 4'], Out: Float[Array, '4 4'], niter: Integer):
    siz = In.shape[0]
    h2 = (1.0 / siz) ** 2
    for it in range(0, niter):
        Out[1:siz - 1, 1:siz - 1] = h2 * (In[0:siz - 2, 1:siz - 1] + In[1:siz - 1, 0:siz - 2] - 4.0 * In[1:siz - 1, 1:siz - 1] + In[2:siz, 1:siz - 1] + In[1:siz - 1, 2:siz])
        In, Out = (Out, In)
