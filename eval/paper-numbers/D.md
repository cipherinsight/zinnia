Task D (folder D/): the running example of the walkthrough.

The paper's walkthrough (zinnia-paper/oopsla-2027-r1/walkthrough.tex and fig/tikz/walkthrough.tex; read
them) compiles this program on the halo2-base backend:

    def kernel(a: Integer[Array, "5 6"], key: Integer[Array, "6"]):  # check the figure for exact types
        p = np.argsort(key, kind="stable"); c = np.empty_like(p); c[p] = np.arange(6)
        im = a[:, c]
        rows = np.flatnonzero(im.any(axis=1)); cols = np.flatnonzero(im.any(axis=0))
        return im[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

(take the exact program from fig/tikz/walkthrough.tex). The text says: "The circuit has X cells, against
Y when each operator is selected on its own." Compute X (default compiler 3ebe0558) and Y (the same with
local selection, env ZINNIA_ABLATE_JOINT=1; see the compiler's docs/ablations.md), as halo2-base advice +
lookup cells (both phases) at the circuit's own parameters, plus Noir gates for both. Also check, with a
mock run on two inputs against NumPy, that the program compiles and outputs NumPy's result, and check
whether the text's claims hold in the compiled circuit: the sort checks a prover-supplied order and
establishes the permutation; the scatter-add's tests [p_k = j] are shared as one-hot vectors; the scans use
running products and their sum-to-one check makes the index assertions (len >= 1) unselected; the window
uses the scans' one-hot vectors. Report which claims hold, with evidence from the selection trace.
