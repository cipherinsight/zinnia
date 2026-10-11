put('pythran_insertion_sort',SIGN+'''
def main(private field[10] list2) -> field[10]:
    for field i in 1..10 do
        field save = list2[i]
        field j = i
        bool active = true
        for u32 step in 0..9 do
            field prev = if j > 0 then j-1 else 0 fi
            bool shift = active && j > 0 && lt(save, list2[prev])
            list2[j] = if shift then list2[prev] else list2[j] fi
            j = if shift then j-1 else j fi
            active = shift
        endfor
        list2[j] = save
    endfor
    return list2
''',2**32,'rewrite',['Replace the while loop by nine guarded shifts; j starts at i<=9 and each iteration decrements j. Preserve insertion sort, without a sorting network.'])
META['pythran_insertion_sort']['as_written']={'status':'not expressible','error':'kernel.py:13: data-dependent while j > 0 and list2[j-1] > save.','constraints':None}
put('pythran_ccopy',SIGN+'''
def main(private field N, private field[4][2] CX, private field INCX, private field[4][2] CY, private field INCY) -> field[4][2]:
    assert(N <= 1024)
    field IX = if lt(INCX,0) then (1-N)*INCX else 0 fi
    field IY = if lt(INCY,0) then (1-N)*INCY else 0 fi
    for field I in 0..1024 do
        bool active = I < N
        field sx = if INCX == 1 && INCY == 1 then I else IX fi
        field sy = if INCX == 1 && INCY == 1 then I else IY fi
        sx = if active then sx else 0 fi
        sy = if active then sy else 0 fi
        sx = if lt(sx,0) then sx+4 else sx fi
        sy = if lt(sy,0) then sy+4 else sy fi
        CY[sy] = if active then CX[sx] else CY[sy] fi
        IX = IX+INCX
        IY = IY+INCY
    endfor
    return CY
''',2**32,'rewrite',['Bound both range(N) loops by 1024 and guard I<N, from @requires(0<=N<=1024). Encode each complex value as real/imaginary fields and expose mutated CY.'])
META['pythran_ccopy']['as_written']={'status':'not expressible','error':'kernel.py:15,24: range(N) has a private bound.','constraints':None}
put('pythran_perm','''
def main(private field[4] iterable) -> field[24][4]:
    field[4] pool = iterable
    field[4] indices = [0,1,2,3]
    field[4] cycles = [4,3,2,1]
    field[24][4] out = [[0; 4]; 24]
    out[0] = iterable
    for u32 step in 1..24 do
        bool done = false
        for u32 rev in 0..4 do
            u32 i = 3-rev
            bool active = !done
            cycles[i] = if active then cycles[i]-1 else cycles[i] fi
            bool rotate = active && cycles[i] == 0
            field old = indices[i]
            for u32 k in 0..3 do
                indices[k] = if rotate && k >= i then indices[k+1] else indices[k] fi
            endfor
            indices[3] = if rotate then old else indices[3] fi
            cycles[i] = if rotate then 4-i else cycles[i] fi
            bool swap = active && !rotate
            field j = if swap then 4-cycles[i] else 0 fi
            field a = indices[i]
            field b = indices[j]
            indices[i] = if swap then b else a fi
            indices[j] = if swap then a else indices[j] fi
            done = done || swap
        endfor
        for u32 k in 0..4 do
            out[step][k] = pool[indices[k]]
        endfor
    endfor
    return out
''',attempt='rewrite',rewrites=['Allocate exactly 4!=24 outputs; replace while/break/for-else with 23 guarded successor iterations after the first output, as implied by the four-element input. Preserve the source cycles algorithm.'])
META['pythran_perm']['as_written']={'status':'not expressible','error':'kernel.py:11,19: growing list and while 1 with break/for-else.','constraints':None}
# A finite domain permits an exhaustive, fixture-independent proof of the Collatz bound.
steps=[]; peak=0
for n in range(3,1025):
 x=n;c=0
 while x != 1:
  x=3*x+1 if x%2 else x//2;c+=1;peak=max(peak,x)
 steps.append(c)
maxsteps=max(steps)
pathlib.Path('pythran_collatz/bound.json').write_text(json.dumps({'domain':[3,1024],'max_transitions':maxsteps,'max_intermediate':peak,'steps':steps},indent=2))
put('pythran_collatz',f'''
def main(private u32 n_max) -> (u32, u32):
    assert(n_max <= 1024)
    u64[1022] n = [0u64; 1022]
    u32[1022] count = [1u32; 1022]
    for u32 i in 0..1022 do
        n[i] = {0}u64
    endfor
    '''+ '\n'.join(f'    n[{i}] = {i+3}u64' for i in range(1022))+f'''
    for u32 step in 0..{maxsteps} do
        for u32 i in 0..1022 do
            bool active = i+3 <= n_max && n[i] != 1
            count[i] = count[i] + if active then 1 else 0 fi
            n[i] = if active then if (n[i] & 1) != 0 then 3*n[i]+1 else n[i]/2 fi else n[i] fi
        endfor
    endfor
    u32 max_i = 0
    u32 best = count[0]
    for u32 i in 1..1022 do
        bool better = i+3 <= n_max && count[i] > best
        max_i = if better then i else max_i fi
        best = if better then count[i] else best fi
    endfor
    return max_i+3, best
''',attempt='rewrite',rewrites=[f'Pad arange to 1022 slots from n_max<=1024; use {maxsteps} guarded Collatz rounds, proved exhaustively for every initial integer 3..1024 in bound.json; maximum intermediate {peak} fits u64.'],types={'n_max':'u32'})
META['pythran_collatz']['as_written']={'status':'not expressible','error':'kernel.py:12-15: private arange extent and data-dependent while np.any(indices).','constraints':None}
put('pythran_laplace','''
struct Output { integer[1024][1024] values; u32 length }
def main(private u32 N, private u32 Niter) -> Output:
    assert(N <= 1024)
    assert(Niter <= 1024)
    integer[1024][1024] u = [[0int; 1024]; 1024]
    for u32 j in 0..1024 do
        u[0][j] = if j < N then 4294967296int else 0int fi
    endfor
    for u32 t in 0..1024 do
        for u32 i in 1..1023 do
            for u32 j in 1..1023 do
                bool active = t < Niter && i+1 < N && j+1 < N
                integer num = ((u[i+1][j]+u[i-1][j])*42949673/4294967296)+((u[i][j+1]+u[i][j-1])*42949673/4294967296)
                integer v = num*4294967296/171798692
                u[i][j] = if active then v else u[i][j] fi
            endfor
        endfor
    endfor
    return Output { values: u, length: N }
''',2**32,'rewrite',['Pad the NxN grid to 1024x1024 and bound time iterations at 1024, guarding all work by N and Niter from the two @requires clauses; expose N for decoding.'],decode='square',types={'N':'u32','Niter':'u32'})
META['pythran_laplace']['as_written']={'status':'not expressible','error':'kernel.py:25-28: private grid dimensions and Niter loop bound.','constraints':None}
# Keep the bounded image loops; constant image zeros imply every color distance is zero.
put('pythran_growcut','''
def main(private u32 N) -> u32:
    assert(N <= 1024)
    field[1024][1024][3] image = [[[0; 3]; 1024]; 1024]
    field[1024][1024][2] state = [[[0; 2]; 1024]; 1024]
    field[1024][1024][2] state_next = state
    state[0][0] = [1,4294967296]
    u32 changes = 0
    for u32 j in 0..1024 do
        for u32 i in 0..1024 do
            field winning_colony = state[i][j][0]
            field defense_strength = state[i][j][1]
            for u32 dj in 0..21 do
                for u32 di in 0..21 do
                    bool valid = i < N && j < N && i+di >= 10 && j+dj >= 10 && i+di < N+10 && j+dj < N+10 && !(di == 10 && dj == 10)
                    u32 ii = if valid then i+di-10 else 0 fi
                    u32 jj = if valid then j+dj-10 else 0 fi
                    // image is identically zero for every input N: sqrt(s)=0, gval=1.
                    field attack_strength = state[ii][jj][1]
                    bool win = valid && attack_strength > defense_strength
                    defense_strength = if win then attack_strength else defense_strength fi
                    winning_colony = if win then state[ii][jj][0] else winning_colony fi
                    changes = changes + if win then 1 else 0 fi
                endfor
            endfor
            state_next[i][j] = [winning_colony,defense_strength]
        endfor
    endfor
    return changes
''',attempt='rewrite',rewrites=['Pad images/state to 1024x1024 using N<=1024 and mask out-of-domain pixels; use a 21x21 masked neighborhood from the literal radius 10.','Fold the input-independent zero color differences: every image entry is initialized to zero and never changed, so sqrt(s)=0 and gval=1 on all valid inputs.'])
META['pythran_growcut']['as_written']={'status':'not expressible','error':'kernel.py:56-58: arrays have private NxN shapes; neighborhood loops also depend on N.','constraints':None}
# FFT twiddles are compile-time constants, not private transcendental operations.
import cmath
fft=''
for size in [1,2,4,8,16]:
 fft+=f'def fft{size}(integer[{size}][2] x) -> integer[{size}][2]:\n'
 if size==1:fft+='    return x\n\n';continue
 half=size//2
 fft+=f'    integer[{half}][2] e = fft{half}(['+', '.join(f'x[{2*k}]' for k in range(half))+'])\n'
 fft+=f'    integer[{half}][2] o = fft{half}(['+', '.join(f'x[{2*k+1}]' for k in range(half))+'])\n'
 fft+=f'    integer[{size}][2] r = [[0int; 2]; {size}]\n'
 for k in range(half):
  w=cmath.exp(-2j*cmath.pi*k/size); re=round(w.real*2**32);im=round(w.imag*2**32)
  fft+=f'    integer tr{k} = (o[{k}][0]*({re}int)-o[{k}][1]*({im}int))/4294967296\n    integer ti{k} = (o[{k}][0]*({im}int)+o[{k}][1]*({re}int))/4294967296\n'
  fft+=f'    r[{k}] = [e[{k}][0]+tr{k},e[{k}][1]+ti{k}]\n    r[{k+half}] = [e[{k}][0]-tr{k},e[{k}][1]-ti{k}]\n'
 fft+='    return r\n\n'
fft+='def main(private integer[16][2] x) -> integer[16][2]:\n    return fft16(x)\n'
put('pythran_fft',fft,2**32)
put('pythran_spectraldns','''
def main(private integer[3][2][5][7][2] dU, private integer[3][2][5][7][2] U_hat, private integer[2][5][7] K2, private integer[3][2][5][7] K, private integer[2][5][7][2] P_hat, private integer[3][2][5][7] K_over_K2, private integer nu) -> (integer[2][5][7][2], integer[3][2][5][7][2]):
    for u32 i in 0..2 do
        for u32 j in 0..5 do
            for u32 k in 0..7 do
                for u32 c in 0..2 do
                    P_hat[i][j][k][c] = dU[0][i][j][k][c]*K_over_K2[0][i][j][k]/4294967296 + dU[1][i][j][k][c]*K_over_K2[1][i][j][k]/4294967296 + dU[2][i][j][k][c]*K_over_K2[2][i][j][k]/4294967296
                    for u32 a in 0..3 do
                        dU[a][i][j][k][c] = dU[a][i][j][k][c] - P_hat[i][j][k][c]*K[a][i][j][k]/4294967296 - (nu*K2[i][j][k]/4294967296)*U_hat[a][i][j][k][c]/4294967296
                    endfor
                endfor
            endfor
        endfor
    endfor
    return P_hat, dU
''',2**32)
