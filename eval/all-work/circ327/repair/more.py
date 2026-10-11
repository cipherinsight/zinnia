from manual import put,change,BASE
import json,pathlib
put('pythran_caxpy','''def main(private u64 N, private u64[2] CA, private u64[4][2] CX, private u64 INCX, private u64[4][2] CY, private u64 INCY) -> u64[9]:
    bool valid = N > 0u64 && (CA[0] != 0u64 || CA[1] != 0u64)
    bool unit = INCX == 1u64 && INCY == 1u64
    u64 ix = if neg(INCX) then (1u64-N)*INCX else 0u64 fi
    u64 iy = if neg(INCY) then (1u64-N)*INCY else 0u64 fi
    for u64 i in 0..1024 do
        bool active = valid && i < N
        u64 x = if unit then i else ix fi
        u64 y = if unit then i else iy fi
        x = if neg(x) then x+4u64 else x fi
        y = if neg(y) then y+4u64 else y fi
        x = if active then x else 0u64 fi
        y = if active then y else 0u64 fi
        u64 re = mulq(CA[0],CX[x][0])-mulq(CA[1],CX[x][1])
        u64 im = mulq(CA[0],CX[x][1])+mulq(CA[1],CX[x][0])
        CY[y][0] = if active then CY[y][0]+re else CY[y][0] fi
        CY[y][1] = if active then CY[y][1]+im else CY[y][1] fi
        ix = ix+INCX
        iy = iy+INCY
    endfor
    return [if valid then 1u64 else 0u64 fi,CY[0][0],CY[0][1],CY[1][0],CY[1][1],CY[2][0],CY[2][1],CY[3][0],CY[3][1]]''',['Guard 1024 source iterations by the explicit 0<=N<=1024 contract, normalize negative valid indices, and expose a return-presence tag for early None returns. Test abs(CA)==0 by both components equaling zero.'])
put('pythran_gauss','''def main(private u64 n, private u64[4][4] a, private u64[4] b) -> u64[4]:
    for u64 i in 1..5 do
        bool active = i <= n
        u64 i0 = i
        u64 amp0 = absv(a[i-1u64][i-1u64])
        for u64 j in 1..5 do
            bool take = active && j > i && j <= n && absv(a[i-1u64][j-1u64]) > amp0
            i0 = if take then j else i0 fi
            amp0 = if take then absv(a[i-1u64][j-1u64]) else amp0 fi
        endfor
        u64 temp = b[i-1u64]
        b[i-1u64] = if active && i < n then b[i0-1u64] else b[i-1u64] fi
        b[i0-1u64] = if active && i < n then temp else b[i0-1u64] fi
        for u64 j in 1..5 do
            u64 t = a[j-1u64][i-1u64]
            bool go = active && i < n && j >= i && j <= n
            a[j-1u64][i-1u64] = if go then a[j-1u64][i0-1u64] else a[j-1u64][i-1u64] fi
            a[j-1u64][i0-1u64] = if go then t else a[j-1u64][i0-1u64] fi
        endfor
        u64 divisor = if active then a[i-1u64][i-1u64] else 4294967296u64 fi
        a[i-1u64][i-1u64] = if active then divq(4294967296u64,divisor) else a[i-1u64][i-1u64] fi
        b[i-1u64] = if active then mulq(b[i-1u64],a[i-1u64][i-1u64]) else b[i-1u64] fi
        for u64 j in 1..5 do
            a[j-1u64][i-1u64] = if active && j > i && j <= n then mulq(a[j-1u64][i-1u64],a[i-1u64][i-1u64]) else a[j-1u64][i-1u64] fi
        endfor
        for u64 k in 1..5 do
            bool go = active && k > i && k <= n
            b[k-1u64] = if go then b[k-1u64]-mulq(a[i-1u64][k-1u64],b[i-1u64]) else b[k-1u64] fi
            for u64 j in 1..5 do
                a[j-1u64][k-1u64] = if go && j > i && j <= n then a[j-1u64][k-1u64]-mulq(a[i-1u64][k-1u64],a[j-1u64][i-1u64]) else a[j-1u64][k-1u64] fi
            endfor
        endfor
    endfor
    for u64 rev in 0..3 do
        u64 i = 3u64-rev
        for u64 j in 1..4 do
            b[i-1u64] = if i < n && j >= i && j < n then b[i-1u64]-mulq(a[j][i-1u64],b[j]) else b[i-1u64] fi
        endfor
    endfor
    return b''',['Bound active iterations by four: any n>4 accesses beyond the declared 4x4 matrix and is not a valid successful input. Guard updates and the reciprocal on inactive iterations.'])
put('pythran_lttb_pt','''def main(private u64[16][2] data, private u64 n_out) -> u64[33]:
    u64[16][2] out = [[0u64; 2]; 16]
    out[0] = data[0]
    out[n_out-1u64] = data[15]
    u64 bins = n_out-2u64
    u64 base = 14u64/bins
    u64 extra = 14u64%bins
    for u64 i in 0..14 do
        bool active = i < bins && n_out < 16u64
        u64 start = 1u64+i*base+(if i < extra then i else extra fi)
        u64 size = base+(if i < extra then 1u64 else 0u64 fi)
        u64 next = start+size
        u64 nsize = if i+1u64 == bins then 1u64 else base+(if i+1u64 < extra then 1u64 else 0u64 fi) fi
        u64[2] mean = [0u64; 2]
        for u64 j in 0..16 do
            bool take = active && j >= next && j < next+nsize
            mean[0] = mean[0]+if take then data[j][0] else 0u64 fi
            mean[1] = mean[1]+if take then data[j][1] else 0u64 fi
        endfor
        mean[0] = divc(mean[0],nsize)
        mean[1] = divc(mean[1],nsize)
        u64 best = 0u64
        u64 chosen = if active then start else 0u64 fi
        for u64 j in 0..16 do
            u64 area = divc(absv(mulq(out[i][0]-mean[0],data[j][1]-out[i][1])-mulq(out[i][0]-data[j][0],mean[1]-out[i][1])),2u64)
            bool take = active && j >= start && j < start+size && (j == start || area > best)
            chosen = if take then j else chosen fi
            best = if take then area else best fi
        endfor
        out[i+1u64] = if active then data[chosen] else out[i+1u64] fi
    endfor
    u64[33] result = [0u64; 33]
    result[0] = n_out
    for u64 i in 0..16 do
        result[1u64+2u64*i] = if n_out == 16u64 then data[i][0] else out[i][0] fi
        result[2u64+2u64*i] = if n_out == 16u64 then data[i][1] else out[i][1] fi
    endfor
    return result''',['Valid returns require 3<=n_out<=16 by explicit source exceptions. Pad the output to sixteen points and guard at most fourteen array_split bins; preserve larger first bins, mean and first argmax tie-breaking.'])
put('npbench_weather_stencils_vadv_vadv','''def main(private u64[15][15][10] utens_stage, private u64[15][15][10] u_stage, private u64[16][15][10] wcon, private u64[15][15][10] u_pos, private u64[15][15][10] utens, private u64 dtr_stage) -> u64[15][15][10]:
    u64[15][15][10] ccol = [[[0u64; 10]; 15]; 15]
    u64[15][15][10] dcol = [[[0u64; 10]; 15]; 15]
    u64[15][15] data_col = [[0u64; 15]; 15]
    for u32 i in 0..15 do
        for u32 j in 0..15 do
            u64 gcv = divc(wcon[i+1u32][j][1]+wcon[i][j][1],4u64)
            u64 cs = divc(gcv,2u64)
            ccol[i][j][0] = divc(gcv,2u64)
            u64 bcol = dtr_stage-ccol[i][j][0]
            u64 correction = 0u64-mulq(cs,u_stage[i][j][1]-u_stage[i][j][0])
            dcol[i][j][0] = mulq(dtr_stage,u_pos[i][j][0])+utens[i][j][0]+utens_stage[i][j][0]+correction
            u64 divided = divq(4294967296u64,bcol)
            ccol[i][j][0] = mulq(ccol[i][j][0],divided)
            dcol[i][j][0] = mulq(dcol[i][j][0],divided)
        endfor
    endfor
    for u32 k in 1..9 do
        for u32 i in 0..15 do
            for u32 j in 0..15 do
                u64 gav = 0u64-divc(wcon[i+1u32][j][k]+wcon[i][j][k],4u64)
                u64 gcv = divc(wcon[i+1u32][j][k+1u32]+wcon[i][j][k+1u32],4u64)
                u64 a_s = divc(gav,2u64)
                u64 cs = divc(gcv,2u64)
                u64 acol = divc(gav,2u64)
                ccol[i][j][k] = divc(gcv,2u64)
                u64 bcol = dtr_stage-acol-ccol[i][j][k]
                u64 correction = 0u64-mulq(a_s,u_stage[i][j][k-1u32]-u_stage[i][j][k])-mulq(cs,u_stage[i][j][k+1u32]-u_stage[i][j][k])
                dcol[i][j][k] = mulq(dtr_stage,u_pos[i][j][k])+utens[i][j][k]+utens_stage[i][j][k]+correction
                u64 divided = divq(4294967296u64,bcol-mulq(ccol[i][j][k-1u32],acol))
                ccol[i][j][k] = mulq(ccol[i][j][k],divided)
                dcol[i][j][k] = mulq(dcol[i][j][k]-mulq(dcol[i][j][k-1u32],acol),divided)
            endfor
        endfor
    endfor
    for u32 i in 0..15 do
        for u32 j in 0..15 do
            u64 gav = 0u64-divc(wcon[i+1u32][j][9]+wcon[i][j][9],4u64)
            u64 a_s = divc(gav,2u64)
            u64 acol = divc(gav,2u64)
            u64 bcol = dtr_stage-acol
            u64 correction = 0u64-mulq(a_s,u_stage[i][j][8]-u_stage[i][j][9])
            dcol[i][j][9] = mulq(dtr_stage,u_pos[i][j][9])+utens[i][j][9]+utens_stage[i][j][9]+correction
            u64 divided = divq(4294967296u64,bcol-mulq(ccol[i][j][8],acol))
            dcol[i][j][9] = mulq(dcol[i][j][9]-mulq(dcol[i][j][8],acol),divided)
            data_col[i][j] = dcol[i][j][9]
            utens_stage[i][j][9] = mulq(dtr_stage,data_col[i][j]-u_pos[i][j][9])
        endfor
    endfor
    for u32 reverse in 0..9 do
        u32 k = 8u32-reverse
        for u32 i in 0..15 do
            for u32 j in 0..15 do
                data_col[i][j] = dcol[i][j][k]-mulq(ccol[i][j][k],data_col[i][j])
                utens_stage[i][j][k] = mulq(dtr_stage,data_col[i][j]-u_pos[i][j][k])
            endfor
        endfor
    endfor
    return utens_stage''')
# Source-grounded blockers, with independent function identities.
def block(n,category,why):
 d=BASE/n;p=json.load(open(d/'entry.json'))
 a={'status':'not_expressible' if category.startswith('not_expressible') else 'needs_gadget','error':why,'constraints':None}
 r={'program':p['program'],'function':p['function'],'category':category,'as_written':a,'rewrite':dict(a,rewrites=[]),'correctness':{x:{'status':'not_run','reason':'No executable translation for this source construct'} for x in ['A','B']},'notes':why,'source':p}
 (d/'result.json').write_text(json.dumps(r,indent=2));(d/'ATTEMPTS.md').write_text(why+'\nNo dummy arithmetic-only program is counted as the kernel.\n')
block('pythran_wdist','needs_gadget: sqrt','kernel.py:18 uses np.sqrt((wdiff ** 2).sum()); native Q32 division repairs the division subexpression but cannot supply square root. All loops already have static bounds.')
block('pythran_log_likelihood','needs_gadget: exp/log','kernel.py:14,16 use numpy.exp(-s) and numpy.log(pdfs). Native unsigned division handles the arithmetic, but these transcendental calls have no native route.')
for f in ['train_eq','train_eq2']:
 block('pythran_train_eq__'+f,'not_expressible: dynamic shape','Function '+f+': err=np.zeros(TrSyms) and range(TrSyms) have no domain-implied bound. os=0 permits a valid ten-tap slice for arbitrarily many TrSyms despite the fixed E shape; output err has TrSyms entries. The explicit arithmetic does not remove this blocker.')
block('npbench_cavity_flow_cavity_flow','not_expressible: dynamic shape','kernel.py:42 b=np.zeros((ny,nx)): nx and ny are unbounded private integers. nt=0 is explicitly valid and permits arbitrarily large b, so the declared 16x16 u/v/p buffers do not bound this intermediate allocation for every valid input. The nt and nit loops themselves can be guarded at 1024; native arithmetic removes their division blocker, but does not bound b as written. Minimal rewrite can eliminate b entirely on nt=0 and use the required 16x16 shape on active executions; see rewrite.zok for that attempt.')
