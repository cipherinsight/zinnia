"""The sizing rule of the acceptance study: small inputs, stated once.

The acceptance (coverage) study asks which programs Zinnia accepts, not how
it scales, and the upstream examples are far too large for any ZK system
(NPBench's smallest preset S multiplies a 1000x1200 by a 1200x1100 matrix).
So every NPBench and Pythran entry is typed and run at a small size:

* every array dimension is at most CAP;
* scalars that are sizes or trip counts are at most CAP (other scalars keep
  their upstream value);
* the relations among dimensions are kept: what is equal stays equal, n and
  n+1 stay n and n+1, a dimension computed from a parameter is still computed
  from it;
* where a loop runs once per permutation of n items (fannkuch, n_queens,
  permutations), n is 4: 4! = 24 iterations, where the upstream 5 to 9 give 120
  to 362880.

Sizes written into a function body are not inputs and stay as upstream wrote
them, since bodies are verbatim: emin's 1,000,000 steps, the 600 x d and
200 x d (500 x d and 200 x d) point sets of allpairs_distances(_loops), guerre's
50-entry work lists, lenet's fully connected widths 120 and 84.

NPBench: preset "T" is preset S with every integer parameter p replaced by
max(min(p, 2), ceil(p / d)), for the smallest whole d such that `initialize()`
succeeds and every integer parameter, integer input and array dimension is at
most CAP. Arrays come from NPBench's own `initialize()` at T, so shapes stay
consistent with each other (spmv's M+1 row pointers and nnz entries, hdiff's
I+4 halo). NPBENCH_T_OVERRIDES lists the exceptions.

Pythran: the upstream `#runas` line (else `#bench`) with its sizes scaled down
by hand to at most CAP, keeping the construction of the line, listed in
PYTHRAN_SMALL_EXAMPLES next to the size it replaces. A line whose arrays and
size scalars are already at most CAP is used as it is. PYTHRAN_NO_UPSTREAM_EXAMPLE
gives an example for the exported functions upstream never calls.
"""

from __future__ import annotations

from typing import Any

CAP = 16

# short_name -> (values that replace the scaled preset, why). A key whose value
# is EXEMPT_DIMS lists array dimensions allowed above CAP.
NPBENCH_T_OVERRIDES: dict[str, tuple[dict[str, Any], str]] = {
    "lenet": ({"N": 2, "H": 16, "W": 16},
              "H and W stay 16: two 5x5 convolutions and two 2x2 poolings need H, W >= 16 for a non-empty feature "
              "map (16 -> 12 -> 6 -> 2 -> 1). The fully connected widths 120, 84 and 10 are constants in "
              "lenet's initialize(), not parameters, so fc1w, fc1b, fc2w, fc2b, fc3w keep dimensions above 16."),
    "nbody": ({"tEnd": 0.8},
              "the trip count Nt = ceil(tEnd / dt) comes from the float tEnd; tEnd = 0.8 (S: 2.0) gives Nt = 16."),
}
# Benchmarks allowed array dimensions above CAP, and why (see NPBENCH_T_OVERRIDES).
NPBENCH_DIMENSION_EXEMPT = {"lenet": {120, 84, 10}}

# (program, function) -> (example line at size <= CAP, what changed).
PYTHRAN_SMALL_EXAMPLES: dict[tuple[str, str], tuple[str, str]] = {
    ("pythran_allpairs", "sqr_dists"): (
        "u = 16; d = 16; import numpy as np; b = np.ones((u,d)); a = np.ones((u,d)); sqr_dists(a, b)",
        "#bench line (no #runas); u = d = 300 -> 16"),
    ("pythran_allpairs_distances", "allpairs_distances"): ("allpairs_distances(16)", "size d 100 -> 16"),
    ("pythran_allpairs_distances_loops", "allpairs_distances_loops"): (
        "allpairs_distances_loops(16)", "size d 100 -> 16"),
    ("pythran_approximated_callgraph", "approximated_callgraph"): (
        "approximated_callgraph(16)", "size 100 -> 16"),
    ("pythran_blacksholes", "BlackScholes"): (
        "BlackScholes(list(range(1,15)), list(range(1,15)), list(range(1,15)), 0.5, 0.76, 12)",
        "option lists range(1,100) -> range(1,15) (99 -> 14 entries); nb_opt = 12 kept (<= 14)"),
    ("pythran_create_grid", "create_grid"): (
        "import numpy as np ; N = 16 ; x = np.arange(0,1,1./N) ; create_grid(x)", "N 1000 -> 16"),
    ("pythran_cronbach", "cronbach"): (
        "import numpy as np ; N = 16 ; items = np.arange(N*N, dtype=float).reshape(N,N) ; cronbach(items)",
        "N 800 -> 16"),
    ("pythran_d2q9_nxnyns", "one_time_step"): (
        "import numpy as np ; r = np.ones((16,16,9)); one_time_step(r,r)",
        "grid 50x50 -> 16x16; the 9 lattice directions of D2Q9 kept"),
    ("pythran_deuxd_convolution", "conv"): (
        "import numpy as np ; x = np.tri(16,16)*0.5 ; w = np.tri(5,5)*0.25 ; conv(x,w)", "image 300x300 -> 16x16"),
    ("pythran_diffusion", "diffuseNumpy"): (
        "import numpy as np;lx,ly=(2**4,2**4);u=np.zeros([lx,ly],dtype=np.double);u[lx//2,ly//2]=1000.0;"
        "tempU=np.zeros([lx,ly],dtype=np.double);diffuseNumpy(u,tempU,16)",
        "grid 2**7 -> 2**4 (impulse still at the centre); trip count iterNum 500 -> 16"),
    ("pythran_euler_challenge14", "euler14"): ("euler14(16)", "trip count MAX_NUM 1000 -> 16"),
    ("pythran_factorize_naive", "factorize_naive"): (
        "factorize_naive(12)",
        "n 12222 -> 12: n bounds the loop, and the kept precondition n <= 1024 rejects 12222; "
        "12 = 2*2*3 still has a repeated factor"),
    ("pythran_fannkuch", "fannkuch2"): (
        "fannkuch(4);fannkuch2(4)", "n 9 -> 4: the loop runs once per permutation (9! = 362880 -> 4! = 24)"),
    ("pythran_fdtd", "fdtd"): (
        "import numpy ; a = numpy.ones((16,16)); fdtd(a,16)", "grid 1000x1000 -> 16x16; trip count steps 20 -> 16"),
    ("pythran_fft", "fft"): (
        "from numpy import ones ; a = ones(2**4, dtype=complex) ; fft(a)", "length 2**10 -> 2**4 (a power of two)"),
    ("pythran_goodexpomeasure", "goodExpoMeasure"): (
        "import numpy as np; n = 16; a = np.arange(n*n*n).reshape((n,n,n)).astype(np.uint8); b = 2. ; "
        "goodExpoMeasure(a, b)", "n 20 -> 16"),
    ("pythran_growcut", "test"): ("test(16)", "size N 50 -> 16"),
    ("pythran_guerre", "guerre"): (
        "guerre([complex(1,2),complex(3,4),complex(5,6),complex(7,8)],2,complex(5.6,4.3),-3.4,16)",
        "trip count nter 20 -> 16"),
    ("pythran_histogram", "histogram"): (
        "histogram([ (i*1.1+j*2.3)%10 for i in range(4) for j in range(4) ],10)",
        "data 100x100 = 10000 values -> 4x4 = 16 values, same formula"),
    ("pythran_hyantes_core", "run"): (
        "run(0.,0.,90.,90., 1., 100., 16, 16, [ [i/10., i/10., i/20.] for i in range(16) ])",
        "range_x = range_y = 80 -> 16 and the 80 points -> 16 (equal stays equal)"),
    ("pythran_l2norm", "l2_norm"): (
        "import numpy as np ; N = 16 ; x = np.arange(N*N, dtype=np.float64).reshape((N,N)) ; l2_norm(x)",
        "N 100 -> 16"),
    ("pythran_lap", "laplacian"): (
        "import numpy as np; var = np.arange(16, dtype=float) ; dh2=.01 ; laplacian(var, dh2)",
        "length 100 -> 16; the line's `from lap import laplacian` is dropped (the function is in the file)"),
    ("pythran_lapl3d", "laplacien"): (
        "import numpy as np ; np.random.seed(0) ; N = 16 ; X = np.random.randn(N,N,3); laplacien(X)",
        "N 500 -> 16; seeded so the example is reproducible (upstream draws without a seed)"),
    ("pythran_laplace", "calc"): ("calc(16,16)", "size N 60 -> 16; trip count Niter 100 -> 16"),
    ("pythran_ln", "ln"): (
        "import numpy; a = numpy.arange(0, 1., 1./16); b = numpy.zeros_like(a) ; ln(a,b)",
        "length 100 -> 16; the output buffer b is zeros_like instead of empty_like (uninitialised memory "
        "is not a reproducible input)"),
    ("pythran_log_likelihood", "log_likelihood"): (
        "import numpy as np ; N = 16 ; a = np.arange(float(N)); b = 0.1; c =1.1; log_likelihood(a, b, c)",
        "N 10000 -> 16"),
    ("pythran_lttb_pt", "downsample"): (
        "import numpy as np; tseries = np.arange(32.).reshape(16, 2); downsample(tseries, 8)",
        "series 100x2 -> 16x2; n_out = 8 kept"),
    ("pythran_make_decision", "md"): (
        "import numpy as np;  s=np.arange(16.)+np.arange(16.)*1.j ; sc = s[::2]; md(s, sc)",
        "length 160 -> 16 (symbols still every other entry, 80 -> 8)"),
    ("pythran_mandel", "mandel"): ("mandel(16,0.,0., 8)", "size 20 -> 16"),
    ("pythran_matmul", "matrix_multiply"): (
        "a=[ [ float(i) for i in range(16)] for j in range(16)] ; matrix_multiply(a,a)", "60x60 -> 16x16"),
    ("pythran_matrix_class_distance", "matrix_class_distance"): (
        "import numpy as np; n = 16;d = 10;iterations = 16;np.random.seed(42);dat = np.random.random((n, d));"
        "dat_filter = np.random.randint(0, n, size=n); dat_points = np.random.random((n, d)); "
        "matrix_class_distance(dat, dat_filter, dat_points, iterations)",
        "n 200 -> 16; trip count iterations 20 -> 16; d = 10 and the seed kept"),
    ("pythran_monte_carlo", "montecarlo_integration"): (
        "montecarlo_integration(1.,10.,16,[x/100. for x in range(16)],16)",
        "numSteps = randsize = 100 -> 16 (equal stays equal), the random table 100 -> 16 entries"),
    ("pythran_morphology", "dilate_decompose_loops"): (
        "import numpy as np ; image = np.tri(8, 16) /2.0 ; dilate_decompose_loops(image, 4)",
        "image 100x200 -> 8x16 (1:2 kept); k = 4 kept"),
    ("pythran_nqueens", "n_queens"): (
        "n_queens(4)", "board 6 -> 4: the search runs once per permutation of the columns (6! = 720 -> 4! = 24); "
        "4 is the smallest board above 1 with a solution"),
    ("pythran_pairwise", "pairwise"): (
        "import numpy as np ; X = np.linspace(0,10,128).reshape(16,8) ; pairwise(X)",
        "X 200x100 -> 16x8 (2:1 kept), same value range"),
    ("pythran_periodic_dist", "dist"): (
        "import numpy as np ; N = 1.6 ; x = np.arange(0., N, 0.1) ; L = 4 ; periodic = True ; "
        "dist(x, x, x, L,periodic, periodic, periodic)",
        "points 200 -> 16 (N 20 -> 1.6 at the same 0.1 spacing); box L = 4 kept"),
    ("pythran_perm", "permutations"): (
        "permutations([1,4,5,6])", "5 items -> 4 (the first four): the loop runs once per permutation "
        "(5! = 120 -> 4! = 24)"),
    ("pythran_primes_sieve", "primes_sieve"): ("primes_sieve(16)", "size limit 100 -> 16"),
    ("pythran_primes_sieve2", "get_primes7"): ("get_primes7(16)", "size n 100 -> 16"),
    ("pythran_rosen", "rosen"): (
        "import numpy as np; r = np.arange(0.,10., .625); rosen(r)", "1000 points on [0, 10) -> 16"),
    ("pythran_sexy_primes", "primes_below"): ("primes_below(16)", "size x 1000 -> 16"),
    ("pythran_slowparts", "slowparts"): (
        "import numpy as np ;d = 8 ;re = 5 ;params = (d, re, np.ones((2*d, d+1, re)), np.ones((d, d+1, re)),  "
        "np.ones((d, 2*d)), np.ones((d, 2*d)), np.ones((d+1, re, d)), np.ones((d+1, re, d)), 1) ; "
        "slowparts(*params)",
        "d 10 -> 8 so that 2*d <= 16; the shapes 2d, d+1, re are still computed from d"),
    ("pythran_smoothing", "smoothing"): (
        "import numpy as np ; a = np.arange(0,1,1./16) ; smoothing(a, .4)", "100 points on [0, 1) -> 16"),
    ("pythran_sobelfilter", "sobelFilter"): (
        "cols = 4; rows = 4 ;image_list=[x%10+y%20 for x in range(cols) for y in range(rows)]; "
        "sobelFilter(image_list, cols, rows)",
        "image 100x100 -> 4x4: the flat image has cols*rows = 16 entries"),
    ("pythran_specialconvolve", "specialconvolve"): (
        "import numpy as np ; r = np.arange(16*16, dtype=np.uint32).reshape(16,16); specialconvolve(r)",
        "100x100 -> 16x16"),
    ("pythran_stone", "whetstone"): ("whetstone(16)", "trip count loopstart 200 -> 16"),
    ("pythran_sum_primes", "sum_primes"): ("sum_primes(16)", "size n 200 -> 16"),
    ("pythran_sumarray3d", "summation"): (
        "import numpy as np ; N=16 ; pos = np.arange(1, N*3. + 1, dtype=np.float32).reshape((N,3)) ; "
        "w = np.ones(N, dtype=np.float32) ; p =  np.arange(N*3., dtype=np.float32).reshape((N,3)) ; "
        "summation(pos, w, p)", "N 30 -> 16"),
    ("pythran_train_eq", "train_eq"): (
        "import numpy as np; np.random.seed(0); s=np.array([1+1j, -1+1j, -1-1j, 1-1j]); "
        "e=np.random.choice(s, (1, 16)); ee=e/np.sqrt(np.mean(abs(e)**2)); "
        "eee=ee+0.02*(np.random.randn(*ee.shape)+1j*np.random.randn(*ee.shape)); r=(np.sqrt(2)/2+0j);"
        "wx=np.zeros((1,10), dtype=eee.dtype); wx[:,4]=1; train_eq(eee.astype(np.complex64), eee.shape[1]-15, 1, "
        "np.float32(1e-3), wx.astype(np.complex64), (np.complex64(r),s.astype(np.complex64)), False)",
        "symbols 100 -> 16; TrSyms is still eee.shape[1] - 15 (85 -> 1); the 10 equaliser taps kept"),
    ("pythran_train_eq", "train_eq2"): (
        "import numpy as np; np.random.seed(0); s=np.array([1+1j, -1+1j, -1-1j, 1-1j]); "
        "e=np.random.choice(s, (1, 16)); ee=e/np.sqrt(np.mean(abs(e)**2)); "
        "eee=ee+0.02*(np.random.randn(*ee.shape)+1j*np.random.randn(*ee.shape)); r=(np.sqrt(2)/2+0j);"
        "wx=np.zeros((1,10), dtype=eee.dtype); wx[:,4]=1; train_eq2(eee.astype(np.complex64), eee.shape[1]-15, 1, "
        "np.float32(1e-3), wx.astype(np.complex64), (np.complex64(r),s.astype(np.complex64)), False)",
        "symbols 100 -> 16; TrSyms is still eee.shape[1] - 15 (85 -> 1); the 10 equaliser taps kept"),
    ("pythran_vibr_energy", "calculate_vibr_energy"): (
        "import numpy as np ; a = np.sin(np.ones(16)) ; b = np.cos(np.ones(16)) ; n = np.arange(16); "
        "calculate_vibr_energy(a, b, n)", "length 1000000 -> 16"),
    ("pythran_wave_simulation", "test"): ("test(16)", "size PARTICLE_COUNT 50 -> 16"),
    ("pythran_zero", "zero"): ("zero(8,16)", "sizes 10, 20 -> 8, 16 (1:2 kept)"),
}

# Exported entry functions that no upstream example calls: (example line, why these values).
PYTHRAN_NO_UPSTREAM_EXAMPLE: dict[tuple[str, str], tuple[str, str]] = {
    ("pythran_allpairs", "sqr_dists_loops"): (
        "u = 16; d = 16; import numpy as np; b = np.ones((u,d)); a = np.ones((u,d)); sqr_dists_loops(a, b)",
        "the sqr_dists example of the same file"),
    ("pythran_fbcorr", "fbcorr"): (
        "import numpy as np; imgs = np.array([[[[i+j+k for i in range(3)] for j in range(8)] for j in range(8)] "
        "for k in range(4)], dtype=float); filters = np.array([[[[i+2*j-k for i in range(3)] for j in range(5)] "
        "for j in range(5)] for k in range(6)], dtype=float); output = np.zeros((4, 6, 4, 4)); "
        "fbcorr(imgs, filters, output)",
        "the formulas of the #runas line of Pythran's fbcorr.py (the list version), at 4 images of 8x8x3; "
        "output is (images, filters, 8-5+1, 8-5+1)"),
    ("pythran_hasting", "fweb"): (
        "import numpy as np; fweb(np.array([0.75, 0.15, 10.0]), 0., 5., 0.1, 3., 2., 0.4, 0.01)",
        "the three-species Hastings-Powell food web (yprime has 3 entries) with its textbook parameters "
        "a1=5, a2=0.1, b1=3, b2=2, d1=0.4, d2=0.01"),
    ("pythran_ldpc_decoder", "Decoding_logBP"): (
        "import numpy as np; H = np.array([[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1],[1,1,0,0,1,1,0,0],"
        "[0,0,1,1,0,0,1,1]]); Bits = [list(np.flatnonzero(row)) for row in H]; "
        "Nodes = [list(np.flatnonzero(col)) for col in H.T]; Lq = np.zeros(H.shape); "
        "Lc = np.array([1.5, -0.5, 2.0, 0.8, -1.2, 0.3, 1.1, -2.0]); Decoding_logBP(H, Bits, Nodes, Lq, Lc, 1)",
        "a regular parity-check matrix with 4 checks of 4 bits each over 8 bits (so Bits and Nodes are "
        "rectangular), zero initial messages, made-up channel log-likelihoods, one iteration (the default)"),
    ("pythran_make_circle_two_points", "_make_circle_one_point"): (
        "_make_circle_one_point([(1.,4.), (0., 8.), (5., 6.)], (3., 4.))",
        "the points of the file's make_circle_two_points #runas line"),
    ("pythran_queens_numba", "solve"): ("solve(4)", "4 queens (the search is on an 8x8 board)"),
}
