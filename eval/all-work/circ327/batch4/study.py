import os,json,pathlib,subprocess,re,ast,math,shutil,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
SRC=pathlib.Path('/home/zhantong/Projects/zinnia-project/wt/literal')
CIRC='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/target/release/examples/'
MEM='/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh'
os.environ.update(TMPDIR=str(ROOT/'tmp'),MEMRUN_MAX='8G',ZSHARP_STDLIB_PATH='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/src/third_party/ZoKrates/zokrates_stdlib/stdlib')
BATCH=json.load(open(ROOT.parent/'batch4.json')); SPECS={}
HELP='''from "EMBED" import u64_to_field
from "EMBED" import u32_to_u64
from "EMBED" import unpack
from "EMBED" import u64_from_bits

def lt(u64 a, u64 b) -> bool:
    return (a ^ 9223372036854775808u64) < (b ^ 9223372036854775808u64)

def mul(u64 a, u64 b) -> u64:
    bool sa = a >= 9223372036854775808u64
    bool sb = b >= 9223372036854775808u64
    u64 aa = if sa then 0u64-a else a fi
    u64 bb = if sb then 0u64-b else b fi
    bool[128] bits = unpack::<128>(u64_to_field(aa)*u64_to_field(bb))
    u64 v = u64_from_bits(bits[32..96])
    return if sa != sb then 0u64-v else v fi

'''
def add(p,code,decode='real',rewrites=None,reason='Direct static-shape translation.'):
 SPECS[p]=dict(code=HELP+code,decode=decode,rewrites=rewrites,reason=reason)
def block(p,cat,reason,asreason=None,rewrites=None):
 SPECS[p]=dict(category=cat,reason=reason,asreason=asreason or reason,rewrites=rewrites)
add('case0335','''def main(private u64[50] a, private u64[50] b, private u64[50] c) -> u64[50]:
    u64[50] result = a
    for u32 i in 0..50 do
        result[i] = if lt(result[i], b[i]) then b[i] else result[i] fi
        result[i] = if lt(result[i], c[i]) then c[i] else result[i] fi
    endfor
    return result
''')
add('case0314','''def main(private u64[8][6] a) -> u32[2]:
    u32 r = 0
    u32 c = 0
    u64 mn = a[0][0]
    for u32 i in 0..8 do
        for u32 j in 0..6 do
            bool larger = lt(a[r][c], a[i][j])
            r = if larger then i else r fi
            c = if larger then j else c fi
            mn = if lt(a[i][j], mn) then a[i][j] else mn fi
        endfor
    endfor
    a[r][c] = mn
    r = 0
    c = 0
    for u32 i in 0..8 do
        for u32 j in 0..6 do
            bool larger = lt(a[r][c], a[i][j])
            r = if larger then i else r fi
            c = if larger then j else c fi
        endfor
    endfor
    return [r,c]
''','int')
add('case0341','''def main(private u64[5][6] X) -> u64[30]:
    u64[30] result = [0u64;30]
    for u32 i in 0..5 do
        for u32 j in 0..6 do
            result[i*6+j] = X[i][j]
        endfor
    endfor
    return result
''','int', ['Preallocate the append-built list to 30 entries and write by position; X has exactly 5*6 elements.'], 'kernel.py:11-13 append-built list; fixed capacity 30 from input shape.')
add('case0322','''def main(private u64[2][2] a) -> (u32[4][2], u32):
    u64 mn = a[0][0]
    for u32 i in 0..2 do
        for u32 j in 0..2 do
            mn = if lt(a[i][j],mn) then a[i][j] else mn fi
        endfor
    endfor
    u32[4][2] result = [[0u32;2];4]
    u32 n = 0
    for u32 i in 0..2 do
        for u32 j in 0..2 do
            bool keep = a[i][j] == mn
            result[n] = if keep then [i,j] else result[n] fi
            n = n + (if keep then 1u32 else 0u32 fi)
        endfor
    endfor
    return result,n
''','pairs', ['Pad argwhere output to 4 coordinate pairs and return its logical length; a 2x2 array has at most 4 matching cells.'], 'kernel.py:11 value-dependent argwhere length.')
add('case0363','''def main(private u64[100] a, private u64 pos, private u64 element) -> u64[101]:
    u64 p = if lt(pos,0u64) then pos+100u64 else pos fi
    u64[101] result = [0u64;101]
    for u64 i in 0u64..101u64 do
        u64 idx = if i > p then i-1u64 else i fi
        idx = if idx < 100u64 then idx else 0u64 fi
        result[i] = if i == p then element*4294967296u64 else a[idx] fi
    endfor
    return result
''')
add('case0370','''def main(private u64[3][3] a) -> bool:
    bool result = true
    for u32 i in 0..3 do
        for u32 j in 0..3 do
            u64 delta = a[i][j]-a[0][j]
            u64 ad = if lt(delta,0u64) then 0u64-delta else delta fi
            u64 ref = if lt(a[0][j],0u64) then 0u64-a[0][j] else a[0][j] fi
            result = result && ad <= mul(42950u64,ref)
        endfor
    endfor
    return result
''','bool')
add('case0399','''def main(private u64[30] A, private u64 a, private u64 b, private u64 c) -> u64[30]:
    u64[30] B = [0u64;30]
    B[0] = a*A[0]
    B[1] = a*A[1]+b*B[0]
    for u32 k in 2..30 do
        B[k] = a*A[k]+b*B[k-1]+c*B[k-2]
    endfor
    return B
''')
add('case0435','''def main(private u64[4][4] a) -> u64[4][4]:
    for u32 j in 0..4 do
        a[1][j] = 0
    endfor
    for u32 i in 0..4 do
        a[i][0] = 0
    endfor
    return a
''','int')
add('case0441','''def main(private u64[4] a, private u64 number) -> bool:
    bool found = false
    for u32 i in 0..4 do
        found = found || a[i] == number
    endfor
    return found
''','bool')
add('case0450','''def main(private u64[7][8][9] a, private u64[2] second, private u64[2] third) -> u64[7][2][2]:
    u64[7][2][2] result = [[[0u64;2];2];7]
    for u32 i in 0..7 do
        for u32 j in 0..2 do
            for u32 k in 0..2 do
                u64 s = if lt(second[j],0u64) then second[j]+8u64 else second[j] fi
                u64 t = if lt(third[k],0u64) then third[k]+9u64 else third[k] fi
                result[i][j][k] = a[i][s][t]
            endfor
        endfor
    endfor
    return result
''')
add('case0471','''def main(private u64[6][2][9][3][7][4] Z) -> u64[6][2][9][3][7][1]:
    u64[6][2][9][3][7][1] result = [[[[[[0u64;1];7];3];9];2];6]
    for u32 a in 0..6 do
      for u32 b in 0..2 do
       for u32 c in 0..9 do
        for u32 d in 0..3 do
         for u32 e in 0..7 do
          result[a][b][c][d][e][0] = Z[a][b][c][d][e][3]
         endfor
        endfor
       endfor
      endfor
    endfor
    return result
''')
add('case0501','''def main(private u64[13][13][2] a, private u64[13][13] b) -> u64[13][13]:
    u64[13][13] result = [[0u64;13];13]
    for u32 i in 0..13 do
        for u32 j in 0..13 do
            u64 k = if lt(b[i][j],0u64) then b[i][j]+2u64 else b[i][j] fi
            result[i][j] = a[i][j][k]
        endfor
    endfor
    return result
''')
add('pythran_multi_export','''def main(private u64 i) -> u64:
    return i
''')
add('pythran_smoothing','''def main(private u64[16] x, private u64 alpha) -> u64[16]:
    u64[16] s = x
    for u32 i in 1..16 do
        s[i] = mul(alpha,x[i])+mul(4294967296u64-alpha,s[i-1])
    endfor
    return s
''')
add('pythran_bubble_sort','''def main(private u64[11] list0) -> u64[11]:
    u64[11] list1 = list0
    for u32 i in 0..10 do
        bool swap_test = false
        for u32 j in 0..(10-i) do
            bool swap = lt(list1[j+1],list1[j])
            u64 temp = list1[j]
            list1[j] = if swap then list1[j+1] else list1[j] fi
            list1[j+1] = if swap then temp else list1[j+1] fi
            swap_test = true
        endfor
    endfor
    return list1
''','int',reason='Source bubble-sort loops, no invented sorting network; break is unreachable because every inner loop executes.')
add('pythran_check_mask','''def main(private bool[3][4] db, private bool[3] out, private u64[3] mask) -> bool[3]:
    for u32 i in 0..3 do
        bool match = true
        for u32 j in 0..3 do
            u64 bit = if db[i][j+1] then 1u64 else 0u64 fi
            match = match && (mask[j] == (mask[j] & bit))
        endfor
        out[i] = out[i] || (match && db[i][0])
    endfor
    return out
''','bool')
add('pythran_deuxd_convolution','''def main(private u64[16][16] x, private u64[5][5] weights) -> u64[16][16]:
    u64[16][16] result = [[0u64;16];16]
    for u32 i in 0..16 do
     for u32 j in 0..16 do
      for u32 ii in 0..5 do
       for u32 jj in 0..5 do
        u32 row = if i+ii < 2 then 0u32 else i+ii-2 fi
        u32 col = if j+jj < 2 then 0u32 else j+jj-2 fi
        row = if row > 5 then 5u32 else row fi
        col = if col > 5 then 5u32 else col fi
        result[i][j] = result[i][j]+mul(x[row][col],weights[ii][jj])
       endfor
      endfor
     endfor
    endfor
    return result
''',reason='Preserves the source clamp bound sw=5, including its unusual use instead of image dimensions.')
add('npbench_polybench_syrk_syrk','''def main(private u64 alpha, private u64 beta, private u64[14][14] C, private u64[14][10] A) -> u64[14][14]:
    for u32 i in 0..14 do
        for u32 j in 0..(i+1) do
            C[i][j] = mul(C[i][j],beta)
        endfor
        for u32 k in 0..10 do
            for u32 j in 0..(i+1) do
                C[i][j] = C[i][j]+mul(mul(alpha,A[i][k]),A[j][k])
            endfor
        endfor
    endfor
    return C
''',reason='Returns the mutated C array (Python function has an implicit None return).')
# Structural and gadget blockers: no surrogate/stub circuit is claimed as a translation.
block('case0298','needs_gadget: sort','kernel.py:11 np.unique sorts values and returns inverse ranks; requires a sorting/unique gadget.', 'kernel.py:11-12 unique-dependent vals.size and output width.', ['At most 5 unique values: pad vals to 5 and b to 5x5 with logical width; sorting remains required.'])
block('case0328','not_expressible: dynamic shape','kernel.py:12 np.pad output length is unbounded length; signature has no upper bound.')
block('case0348','needs_gadget: matrix_rank','kernel.py:18 np.linalg.matrix_rank requires numerical rank/SVD machinery.', 'kernel.py:12-20 LI and tmp are value-dependent lists.', ['At most 5 independent rows (tmp capacity 6); pad lists and guard by length. Numerical rank remains required.'])
block('case0357','needs_gadget: sort','kernel.py:11 np.argsort requires sort indices; no stock Z# argsort.')
block('case0383','needs_gadget: sort','kernel.py:11 np.argsort requires sort indices.', 'kernel.py:11 slice [:N] has input-dependent length.', ['Pad sliced output to 100 entries with Python slice length; input a has 100 elements. Sorting remains required.'])
block('case0406','not_expressible: dynamic shape','kernel.py:12 output length max(index)+1 has no bound in declared input domain; unique also needs sorting.')
block('case0419','needs_gadget: division','kernel.py:12 mean over private bin_size requires signed fixed-point division.', 'kernel.py:12 reshape has private-dependent bin count.', ['Input width 6 bounds valid positive bin_size and output width by 6; pad bins and guard. Fixed-point division remains required.'])
block('case0428','needs_gadget: sqrt','kernel.py:11 np.std requires fixed-point square root (and mean division).')
block('case0457','needs_gadget: sqrt','kernel.py:11 np.linalg.norm requires fixed-point square root.')
block('case0482','needs_gadget: least_squares','kernel.py:11 np.polyfit requires numerical least-squares/SVD and fixed-point division.')
block('case0495','not_expressible: dynamic shape','kernel.py:11 private shape controls unbounded padded output dimensions.')
block('npbench_channel_flow_channel_flow','not_expressible: dynamic loop','kernel.py:36 convergence while udiff>0.001 has no domain-implied iteration bound (nit bounds only the inner solver).')
block('npbench_deep_learning_mlp_mlp','needs_gadget: exp','kernel.py:15 softmax np.exp requires fixed-point exp; line 17 also requires division.')
block('npbench_nbody_nbody','not_expressible: dynamic shape','kernel.py:61-62 energy arrays have length Nt+1 and line 65 loop uses unbounded Nt; bound on N does not bound Nt.')
block('npbench_polybench_correlation_correlation','needs_gadget: sqrt','kernel.py:11,14 standard deviation and sqrt(float_n) require square root and division.', 'kernel.py:15 np.eye(M) and line 16 loop depend on private M.', ['Valid indexed columns imply M<=14 except degenerate nonpositive sizes; capacity 14 and guarded loops suffice; sqrt remains required.'])
block('npbench_polybench_heat_3d_heat_3d','not_expressible: dynamic loop','kernel.py:10 range(1,TSTEPS) has no declared upper bound and no index-implied bound on TSTEPS.')
block('npbench_polybench_ludcmp_ludcmp','needs_gadget: division','kernel.py:15,21 division by private matrix entries requires fixed-point division.')
block('npbench_stockham_fft_stockham_fft','needs_gadget: exp','kernel.py:13,20 complex exponential requires sin/cos gadgets.', 'kernel.py:11-12 arrays depend on private R; line 16 loop depends on K.', ['For K>=1, reshape validity R**K=16 bounds R<=16 and K<=4; K<=0 skips the loop but unbounded R still allocates the unused DFT. No complete bounded rewrite; complex exponential also missing.'])
block('pythran_allpairs_distances_loops','not_expressible: dynamic shape','kernel.py:19-20 ones((500,d)), ones((200,d)) use unbounded d; do not algebraically replace the kernel by zeros.')
block('pythran_convnet','needs_gadget: tanh','kernel.py:18 tanh and helper line 10 exp/division require gadgets.', 'kernel.py:17,20 private-dependent reshape and allocations.', ['Nonempty valid reshape fixes length*batch_size*state_size=4, bounding positive dimensions by 4; pad and guard. tanh/exp remain required.'])
block('pythran_gauss','needs_gadget: division','kernel.py:32 reciprocal of a private pivot requires fixed-point division.', 'kernel.py:29 private n loop bound.', ['Guard loops over 4 rows; valid indexed n cannot exceed 4, and the explicit domain also gives n<=1024. Reciprocal remains required.'])
block('pythran_hasting','needs_gadget: division','kernel.py:11-13 private denominators require fixed-point division.')
block('pythran_lap','needs_gadget: division','kernel.py:25 lap/dh2 requires fixed-point division by private dh2.')
block('pythran_log_likelihood','needs_gadget: division','kernel.py:11 private fixed-point division; lines 12-14 also require exp/log.')
block('pythran_mandel','not_expressible: dynamic loop','kernel.py:18 while loop has private max_iteration without upper bound; size<=1024 bounds only image shape.', 'kernel.py:11 output shape depends on size; line 18 while loop depends on max_iteration.', ['Pad image to 1024x1024 using the size contract; no finite iteration bound follows for max_iteration.'])
block('pythran_pairwise','needs_gadget: sqrt','kernel.py:18 np.sqrt(d) requires fixed-point square root.')
block('pythran_primes_sieve2','needs_gadget: sqrt','kernel.py:19 n**0.5 requires square root; do not replace the source algorithm by a different sieve.', 'kernel.py:18 dynamic list and lines 23,27 while loops.', ['n<=1024 gives capacity 511 odd candidates, at most 511 outer steps and 511 inner steps; pad and guard. Square root remains required.'])
block('pythran_repeating','not_expressible: dynamic shape','kernel.py:11 output length 10*(1+nvar_y) has no upper bound in the signature.')
block('pythran_sum_primes','needs_gadget: sqrt','kernel.py:15 math.sqrt(n) and ceil require a square-root gadget.', 'kernel.py:27 value-dependent range and filtered list; helper line 17 while.', ['n<=1024 bounds candidate and divisor loops by 1024; mask the sum instead of building the filtered list. Square root remains required.'])
block('pythran_vibr_energy','needs_gadget: exp','kernel.py:10 numpy.exp requires fixed-point exponential.')
add('case0304','''def main(private u64[23] A, private u64 ncol) -> (u64[23],u64,u64):
    u64 col = (23u64/ncol)*ncol
    u64[23] result = [0u64;23]
    for u64 i in 0u64..23u64 do
        result[i] = if i < col then A[22u64-i] else 0u64 fi
    endfor
    return result,23u64/ncol,ncol
''','flatshape', ['Store the reversed slice in capacity 23 with logical row/column dimensions; the slice never contains more than the 23 input values, even when ncol>23 gives zero rows.'], 'kernel.py:12-13 private-dependent slice and reshape.')
add('case0389','''def main(private u64[6][3][4] a, private u64 h, private u64 w) -> (u64[72],u64,u64):
    u64 groups = h/3u64
    u64 tiles = 6u64/groups
    u64[72] result = [0u64;72]
    for u64 k in 0u64..72u64 do
        u64 tile = (k/(tiles*12u64))*tiles+(k/4u64)%tiles
        u64 row = (k/(tiles*4u64))%3u64
        result[k] = a[tile][row][k%4u64]
    endfor
    return result,h,72u64/h
''','flatshape', ['Use a fixed 72-element buffer with logical output dimensions and compute the reshape/swapaxes address; reshapes preserve all 6*3*4 elements. Infer w=-1 as NumPy does.'], 'kernel.py:12 private-dependent reshape dimensions.')
add('case0413','''def main(private u64[10][2] x) -> (u64[10][2],u32):
    u64[10][2] result = [[0u64;2];10]
    u32 n = 0
    for u32 i in 0..10 do
        bool keep = x[i][1] != 0u64
        result[n] = if keep then x[i] else result[n] fi
        n = n+(if keep then 1u32 else 0u32 fi)
    endfor
    return result,n
''','complex', ['Encode each complex number as a pair of Q32 integers; pad filtered output to 10 pairs and return length, bounded by the input length.'], 'kernel.py:11 boolean filter has value-dependent length.')
add('case0508','''def main(private u64[10][10] A) -> (u64[10][10],u32,u32):
    u32 ys = 10
    u32 xs = 10
    u32 ye = 0
    u32 xe = 0
    for u32 i in 0..10 do
        for u32 j in 0..10 do
            bool nz = A[i][j] != 0u64
            ys = if nz && i < ys then i else ys fi
            xs = if nz && j < xs then j else xs fi
            ye = if nz && i+1 > ye then i+1 else ye fi
            xe = if nz && j+1 > xe then j+1 else xe fi
        endfor
    endfor
    u32 rows = ye-ys
    u32 cols = xe-xs
    u64[10][10] result = [[0u64;10];10]
    for u32 i in 0..10 do
        for u32 j in 0..10 do
            bool keep = i < rows && j < cols
            u32 r = if keep then i+ys else 0u32 fi
            u32 c = if keep then j+xs else 0u32 fi
            result[i][j] = if keep then A[r][c] else 0u64 fi
        endfor
    endfor
    return result,rows,cols
''','crop', ['Fuse argwhere coordinate materialization with guarded min/max reductions; pad cropped output to 10x10 plus logical dimensions, bounded by the input shape. All-zero input is invalid in NumPy (empty min).'], 'kernel.py:11 argwhere and line 13 crop have data-dependent shapes.')
add('npbench_polybench_fdtd_2d_fdtd_2d','''def main(private u64 TMAX, private u64[15][16] ex, private u64[15][16] ey, private u64[15][16] hz, private u64[2] _fict_) -> (u64[15][16],u64[15][16],u64[15][16]):
    for u64 t in 0u64..2u64 do
        bool active = lt(t,TMAX)
        for u32 j in 0..16 do
            ey[0][j] = if active then _fict_[t] else ey[0][j] fi
        endfor
        for u32 i in 1..15 do
            for u32 j in 0..16 do
                ey[i][j] = if active then ey[i][j]-mul(2147483648u64,hz[i][j]-hz[i-1][j]) else ey[i][j] fi
            endfor
        endfor
        for u32 i in 0..15 do
            for u32 j in 1..16 do
                ex[i][j] = if active then ex[i][j]-mul(2147483648u64,hz[i][j]-hz[i][j-1]) else ex[i][j] fi
            endfor
        endfor
        for u32 i in 0..14 do
            for u32 j in 0..15 do
                hz[i][j] = if active then hz[i][j]-mul(3006477107u64,ex[i][j+1]-ex[i][j]+ey[i+1][j]-ey[i][j]) else hz[i][j] fi
            endfor
        endfor
    endfor
    return ex,ey,hz
''','fdtd', ['Bound time loop by 2 and guard t<TMAX; every valid positive iteration indexes _fict_[t] of length 2. Negative TMAX executes zero iterations. Return all three mutated arrays.'], 'kernel.py:10 private-dependent time loop.')
# A capacity exists, but the faithful permutation enumeration has an enormous bound.
fannbound=str(math.factorial(1024))
add('pythran_fannkuch',f'''def main(private u64 n) -> bool:
    u64[1024] count = [0u64;1024]
    u64[1024] perm1 = [0u64;1024]
    u64[1024] perm = [0u64;1024]
    for u64 i in 0u64..1024u64 do
        count[i] = i+1u64
        perm1[i] = i
        perm[i] = i
    endfor
    u64 max_flips = 0
    u64 m = n-1u64
    u64 r = n
    u64 check = 0
    bool active = true
    for integer step in 0int..{fannbound}int do
        check = if active && check < 30u64 then check+1u64 else check fi
        for u64 t in 0u64..1024u64 do
            bool go = active && r != 1u64
            u64 idx = if go then r-1u64 else 0u64 fi
            count[idx] = if go then r else count[idx] fi
            r = if go then r-1u64 else r fi
        endfor
        bool flip = active && perm1[0] != 0u64 && perm1[m] != m
        perm = if flip then perm1 else perm fi
        u64 flips_count = 0
        u64 k = perm[0]
        for integer f in 0int..{fannbound}int do
            bool go = flip && k != 0u64
            u64[1024] old = perm
            for u64 j in 0u64..1024u64 do
                u64 idx = if go && j <= k then k-j else j fi
                perm[j] = if go && j <= k then old[idx] else old[j] fi
            endfor
            flips_count = flips_count+(if go then 1u64 else 0u64 fi)
            k = perm[0]
        endfor
        max_flips = if flip && flips_count > max_flips then flips_count else max_flips fi
        bool carry = active
        for u64 t in 0u64..1024u64 do
            bool go = carry && r != n
            u64[1024] old = perm1
            for u64 j in 0u64..1024u64 do
                u64 idx = if j < r then j+1u64 else 0u64 fi
                idx = if idx < 1024u64 then idx else 0u64 fi
                perm1[j] = if go && j < r then old[idx] else (if go && j == r then old[0] else old[j] fi) fi
            endfor
            u64 idx = if go then r else 0u64 fi
            count[idx] = if go then count[idx]-1u64 else count[idx] fi
            bool stop = go && count[idx] > 0u64
            r = if go && !stop then r+1u64 else r fi
            carry = carry && !stop
        endfor
        active = active && !carry
    endfor
    return true
''','none', ['Pad count/perm buffers to 1024 per explicit n<=1024 contract; guard every update after early returns/breaks.', 'Bound permutation enumeration and each terminating prefix-reversal chain by 1024! (distinct permutations), and carry/descent loops by 1024; preserve call although fannkuch2 returns None.'], 'kernel.py:9,14-16 dynamic lists and nested while loops; bounded rewrite needs factorial-sized unrolling.')
SPECS['npbench_stockham_fft_stockham_fft']['category']='not_expressible: dynamic shape'
SPECS['npbench_stockham_fft_stockham_fft']['reason']='kernel.py:11-12 allocate R by R with unbounded R when K=0 (valid, skips all FFT iterations); complex exp also needs a gadget.'

# Mean by an integer count uses stock integer division: no fixed-point divider gadget.
add('case0419','''def main(private u64[8][6] data, private u64 bin_size) -> u64[50]:
    u64 cols = 6u64/bin_size
    u64[50] packed = [0u64;50]
    for u64 i in 0u64..8u64 do
        for u64 j in 0u64..6u64 do
            bool keep = j < cols
            u64 b = if keep then cols-1u64-j else 0u64 fi
            u64 sum = 0
            for u64 k in 0u64..6u64 do
                bool active = keep && k < bin_size
                u64 idx = if active then 5u64-b*bin_size-k else 0u64 fi
                sum = sum+(if active then data[i][idx] else 0u64 fi)
            endfor
            bool negative = lt(sum,0u64)
            u64 magnitude = if negative then 0u64-sum else sum fi
            u64 mean = magnitude/bin_size
            packed[i*6u64+j] = if negative then 0u64-mean else mean fi
        endfor
    endfor
    packed[48] = 8
    packed[49] = cols
    return packed
''','bins', ['Pad the output to 8x6 and publish its logical column count; each bin consumes at least one of 6 input columns, while bin_size>6 gives zero bins.', 'Replace the private reshape with guarded address arithmetic; divide the signed Q32 sum by its integer count using stock u64 division (no rescaling and no handwritten division circuit).'], 'kernel.py:12 reshape/bin count is private-dependent; capacity 8x6 from input shape.')

# Pack multi-output records because this Z# frontend allows exactly one return value.
for name,size in [('case0304',23),('case0389',72)]:
 sp=SPECS[name]
 sp['code']=sp['code'].replace(f'-> (u64[{size}],u64,u64):',f'-> u64[{size+2}]:')
 dims=('23u64/ncol','ncol') if name=='case0304' else ('h','72u64/h')
 old='return result,'+','.join(dims)
 sp['code']=sp['code'].replace(old,f'u64[{size+2}] packed = [0u64;{size+2}]\n    for u32 i in 0..{size} do\n        packed[i] = result[i]\n    endfor\n    packed[{size}] = {dims[0]}\n    packed[{size+1}] = {dims[1]}\n    return packed')
 sp['rewrites'].append('Pack the buffer and logical dimensions into one public return array (Z# permits one return value).')
for name,typ,rows,cols in [('case0322','u32',4,2),('case0413','u64',10,2),('case0508','u64',10,10)]:
 sp=SPECS[name];n=rows*cols;crop=name=='case0508';total=n+(2 if crop else 1)
 sp['code']=re.sub(r'-> \(.*?\):',f'-> {typ}[{total}]:',sp['code'])
 old='return result,rows,cols' if crop else 'return result,n'
 packing=f'{typ}[{total}] packed = [0{typ};{total}]\n    for u32 i in 0..{rows} do\n        for u32 j in 0..{cols} do\n            packed[i*{cols}+j] = result[i][j]\n        endfor\n    endfor\n'
 cast=lambda x: 'u32_to_u64('+x+')' if typ=='u64' else x
 packing+=f'    packed[{n}] = '+cast('rows' if crop else 'n')+'\n'
 if crop:packing+=f'    packed[{n+1}] = u32_to_u64(cols)\n'
 sp['code']=sp['code'].replace(old,packing+'    return packed')
 sp['rewrites'].append('Pack buffer and length/dimensions into one public return array.')
sp=SPECS['npbench_polybench_fdtd_2d_fdtd_2d']
sp['code']=sp['code'].replace('_fict_', 'fict').replace('-> (u64[15][16],u64[15][16],u64[15][16]):','-> u64[3][15][16]:').replace('return ex,ey,hz','return [ex,ey,hz]')
SPECS['case0341']['rewrites']=None
SPECS['case0341']['reason']='kernel.py:11-13 list length is statically 30 (5x6), translated as a fixed array.'

def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def run_cmd(cmd,log):
 start=time.monotonic()
 p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 log.write_text(p.stdout)
 return p.returncode,p.stdout,time.monotonic()-start

def fixture_args(f,label):
 args={}
 for prm in f['parameters']:
  name=prm['name']; v=f['inputs'][label][name]; dtype=prm.get('dtype','int')
  if dtype=='complex':
   ar=np.array(v); v=ar[...,0]+1j*ar[...,1]
  elif prm['kind']=='array': v=np.array(v,dtype={'int':np.int64,'float':np.float64,'bool':np.bool_}[dtype])
  elif prm['kind']=='tuple': v=tuple(v)
  args[name]=v
 return args

def pin_text(f,label):
 leaves=[]
 def rec(name,v,dtype):
  if isinstance(v,list):
   for i,x in enumerate(v): rec(name+'.'+str(i),x,dtype)
  else:
   val=('true' if v else 'false') if dtype=='bool' else '#x'+format((int(round(v*2**32)) if dtype in ('float','complex') else int(v))%(2**64),'016x')
   leaves.append(f'({name} {val})')
 for p in f['parameters']: rec('fict' if p['name']=='_fict_' else p['name'],f['inputs'][label][p['name']],p.get('dtype','int'))
 return '(let (\n'+'\n'.join(leaves)+'\n) false)\n'

def reference(src,fn,args,p):
 tree=ast.parse(src)
 tree.body=[x for x in tree.body if not (isinstance(x,ast.ImportFrom) and x.module=='zinnia')]
 for x in ast.walk(tree):
  if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)):
   x.decorator_list=[];x.returns=None
   for a in x.args.args:a.annotation=None
 ns={};exec(compile(ast.fix_missing_locations(tree),'<numpy-reference>','exec'),ns)
 out=ns[fn](**args)
 if p=='npbench_polybench_syrk_syrk':return args['C']
 if p=='npbench_polybench_fdtd_2d_fdtd_2d':return (args['ex'],args['ey'],args['hz'])
 return out

def parse_output(s):
 # zxi pretty prints nested arrays and tuples, integers as hexadecimal.
 s=s.strip(); s=re.sub(r'\btrue\b','True',s);s=re.sub(r'\bfalse\b','False',s)
 return ast.literal_eval(s)
def signed(v):
 a=np.asarray(v,dtype=object)
 return np.vectorize(lambda x:int(x)-2**64 if int(x)>=2**63 else int(x),otypes=[object])(a)
def decode(raw,mode):
 if mode in ('real','int','bool'):return np.asarray(signed(raw),dtype=float)/(2**32 if mode=='real' else 1)
 if mode=='pairs':return np.asarray(raw[:-1]).reshape(4,2)[:raw[-1]]
 if mode=='complex':
  a=np.asarray(signed(raw[:-1]),dtype=float).reshape(10,2)[:raw[-1]]/2**32;return a[:,0]+1j*a[:,1]
 if mode=='flatshape':
  data,rows,cols=raw[:-2],raw[-2],raw[-1];return np.asarray(signed(data[:rows*cols]),dtype=float).reshape(rows,cols)/2**32
 if mode=='crop':return np.asarray(signed(raw[:-2]),dtype=float).reshape(10,10)[:raw[-2],:raw[-1]]
 if mode=='bins':return np.asarray(signed(raw[:48]),dtype=float).reshape(8,6)[:,:raw[49]]/2**32
 if mode=='fdtd':return np.asarray(signed(raw),dtype=float)/2**32
 if mode=='none':return None

def serial(v):
 if isinstance(v,np.ndarray):return serial(v.tolist())
 if isinstance(v,(tuple,list)):return [serial(x) for x in v]
 if isinstance(v,complex):return [v.real,v.imag]
 if isinstance(v,np.generic):return v.item()
 return v

def execute(p):
 name=p['program'];sp=SPECS[name];d=ROOT/name;d.mkdir(exist_ok=True)
 src=(SRC/p['kernel']).read_text(); (d/'source.py').write_text(src)
 f=json.load(open(SRC/pathlib.Path(p['kernel']).parent/'fixture.json'));write_json(d/'fixture.json',f)
 result=dict(program=name,category=None,as_written=None,rewrite=None,correctness={'A':None,'B':None},notes=sp['reason'])
 if 'code' not in sp:
  status='not_expressible' if ('not_expressible:' in sp['category'] or sp['asreason']!=sp['reason']) else 'needs_gadget'
  result['category']=sp['category'];result['as_written']=dict(status=status,error=sp['asreason'],constraints=None)
  result['rewrite']=dict(status='not_expressible' if sp['category'].startswith('not_expressible:') else 'needs_gadget',error=sp['reason'],constraints=None,rewrites=sp.get('rewrites') or ['No allowed local rewrite removes this obstruction without changing the domain or implementing a prohibited gadget.'])
  (d/'attempts.md').write_text('As written: '+sp['asreason']+'\n\nMinimal rewrite: '+sp['reason']+'\n\n'+'\n'.join(result['rewrite']['rewrites'])+'\n\nNo compilable translation exists under the brief for these attempts; no stub was compiled.\n')
  write_json(d/'result.json',result);print(name,result['category'],flush=True);return
 attempt='rewrite' if sp['rewrites'] else 'as_written'
 if sp['rewrites']:result['as_written']=dict(status='not_expressible',error=sp['reason'],constraints=None)
 path=d/(attempt+'.zok');path.write_text('// '+sp['reason']+'\n'+sp['code'])
 cmd=[MEM,'timeout','300',CIRC+'circ','--language','zsharp',str(path),'r1cs','--action','count']
 write_json(d/'command.json',dict(argv=cmd,env={k:os.environ[k] for k in ['TMPDIR','MEMRUN_MAX','ZSHARP_STDLIB_PATH']}))
 rc,out,elapsed=run_cmd(cmd,d/(attempt+'.compile.log'))
 m=re.search(r'Final r1cs: (\d+) constraints',out)
 status='accepted' if rc==0 and m else 'timeout' if rc==124 else 'out_of_memory' if rc in (137,-9) else 'rejected'
 att=dict(status=status,error=None if status=='accepted' else '\n'.join(out.splitlines()[:18]),constraints=int(m[1]) if m else None,elapsed_seconds=round(elapsed,3),exit_code=rc)
 if sp['rewrites']:att['rewrites']=sp['rewrites']
 result[attempt]=att
 result['category']=('accepted_with_rewrite' if sp['rewrites'] else 'accepted_as_written') if status=='accepted' else 'timeout_or_oom' if status in ('timeout','out_of_memory') else 'circ_error'
 if status=='accepted':
  for label in ['A','B']:
   pin=d/(label+'.pin');pin.write_text(pin_text(f,label))
   rc,txt,secs=run_cmd([MEM,'timeout','300',CIRC+'zxi',str(path),str(pin)],d/(label+'.zxi.log'))
   try:
    if rc:raise RuntimeError(txt[:1600])
    raw=parse_output(txt);got=decode(raw,sp['decode']);want=reference(src,p['function'],fixture_args(f,label),name)
    match=(got is None and want is None) or (np.shape(got)==np.shape(want) and bool(np.allclose(got,want,rtol=1e-6,atol=1e-6)))
    err=float(np.max(np.abs(np.asarray(got)-np.asarray(want)))) if got is not None and np.size(got) else 0.
    result['correctness'][label]=dict(status='pass' if match else 'wrong',max_absolute_error=err)
    write_json(d/(label+'.comparison.json'),dict(expected=serial(want),actual=serial(got),match=match,rtol=1e-6,atol=1e-6))
   except Exception as e:result['correctness'][label]=dict(status='error',error=str(e))
  if not all(v['status']=='pass' for v in result['correctness'].values()):
   att['status']='wrong';result['category']='wrong'
 write_json(d/'result.json',result);print(name,result['category'],att['constraints'],result['correctness'],flush=True)

if __name__=='__main__':
 import sys
 selected=set(sys.argv[1:])
 missing=[p['program'] for p in BATCH if p['program'] not in SPECS]
 assert not missing,missing
 for p in BATCH:
  if not selected or p['program'] in selected: execute(p)
