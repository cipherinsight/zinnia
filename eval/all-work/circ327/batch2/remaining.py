from numeric_translations import *
spec('pythran_morphology','''def main(private field[8][16] x, private field k) -> field[8][16]:
    u32 radius = idx(k)/2
    field[8][16] y = [[0; 16]; 8]
    for u32 i in 0..8 do
        for u32 j in 0..16 do
            u32 left = if i > radius then i-radius else 0 fi
            u32 right = if i+radius+1 < 8 then i+radius+1 else 8 fi
            field currmax = x[left][j]
            for u32 ii in 0..8 do
                currmax = if ii > left && ii < right && lt(currmax,x[ii][j]) then x[ii][j] else currmax fi
            endfor
            y[i][j] = currmax
        endfor
    endfor
    field[8][16] z = [[0; 16]; 8]
    for u32 i in 0..8 do
        for u32 j in 0..16 do
            u32 left = if j > radius then j-radius else 0 fi
            u32 right = if j+radius+1 < 16 then j+radius+1 else 16 fi
            field currmax = y[i][left]
            for u32 jj in 0..16 do
                currmax = if jj > left && jj < right && lt(currmax,y[i][jj]) then y[i][jj] else currmax fi
            endfor
            z[i][j] = currmax
        endfor
    endfor
    return z
''',rewrite=['Replace private inner loop ranges with 8- and 16-iteration guarded scans.'],reason='kernel.py:18,29: private loop endpoints; image dimensions imply bounds 8 and 16, and requires bounds k to 0..1024.')
spec('pythran_nd_local_maxima','''def main(private field[8][6][4][2] data) -> bool[8][6][4][2]:
    bool[8][6][4][2] result = [[[[true; 2]; 4]; 6]; 8]
    for u32 i in 0..8 do
        for u32 j in 0..6 do
            for u32 k in 0..4 do
                for u32 l in 0..2 do
                    field myval = data[i][j][k][l]
                    for u32 oi in 0..8 do
                        for u32 oj in 0..6 do
                            for u32 ok in 0..4 do
                                for u32 ol in 0..2 do
                                    result[i][j][k][l] = result[i][j][k][l] && !lt(myval,data[(i+oi+4)%8][(j+oj+3)%6][(k+ok+2)%4][(l+ol+1)%2])
                                endfor
                            endfor
                        endfor
                    endfor
                endfor
            endfor
        endfor
    endfor
    return result
''','bool',reason='Default mode=wrap; all static np.ndindex positions and offsets expanded, retaining source shape and all comparisons.')
# Ragged windows have at most 4x4 values each, and exactly 16 windows.
spec('case0467','''def half(field x) -> field:
    field mag = ab(x)
    u32 small = idx(if lt(mag, 32) then mag else 32 fi)
    field q = u32_to_field(small/2)
    return if neg(x) then -q-u32_to_field(small%2) else q fi

def start(field x) -> field:
    return if neg(x) then 0 else if lt(4,x) then 4 else x fi fi

def endpoint(field x) -> field:
    field y = if neg(x) then x+4 else x fi
    return start(y)

def main(private field[4][4] a, private field[2] size) -> field[16][18]:
    field rwin = half(size[0])
    field cwin = half(size[1])
    field[16][18] result = [[0; 18]; 16]
    for u32 i in 0..4 do
        field xmin = start(u32_to_field(i)-rwin)
        field xmax = endpoint(if lt(u32_to_field(i)+rwin+1,4) then u32_to_field(i)+rwin+1 else 4 fi)
        field nr = if lt(xmin,xmax) then xmax-xmin else 0 fi
        for u32 j in 0..4 do
            field ymin = start(u32_to_field(j)-cwin)
            field ymax = endpoint(if lt(u32_to_field(j)+cwin+1,4) then u32_to_field(j)+cwin+1 else 4 fi)
            field nc = if lt(ymin,ymax) then ymax-ymin else 0 fi
            result[i*4+j][0] = nr
            result[i*4+j][1] = nc
            for u32 r in 0..4 do
                for u32 c in 0..4 do
                    result[i*4+j][2+r*4+c] = if lt(u32_to_field(r),nr) && lt(u32_to_field(c),nc) then a[idx(xmin+u32_to_field(r))][idx(ymin+u32_to_field(c))] else 0 fi
                endfor
            endfor
        endfor
    endfor
    return result
''','int',rewrite=['Pad each of 16 windows to 4x4 and retain its two dimensions.','Clamp radius magnitude at 16 before endpoint normalization: beyond it all positive-radius windows are full and negative-radius windows empty.'],reason='kernel.py:22: private slice sizes make a ragged list; each slice has at most 4x4 elements from the declared source shape.')
# Fibonacci rewrite uses field arithmetic; F(1024) exceeds field, so native integer
# must be kept rather than silently wrapping or narrowing n to the fixtures.
spec('pythran_fibo',INTHELP+'''def main(private integer n) -> integer:
    integer a = 0
    integer b = 1
    for u32 i in 0..1024 do
        integer next = a+b
        a = if it(i) < n then b else a fi
        b = if it(i) < n then next else b fi
    endfor
    return a
''','int',rewrite=['Replace recursive Fibonacci evaluation by the equivalent recurrence with 1024 guarded iterations.'],reason='kernel.py:12-13,23-24: input-dependent recursion; @requires n<=1024 permits bounded recurrence. Preserve arbitrary integer result (F1024 exceeds a field).',helpers=False)
# Complex arithmetic lowered to pairs, while private trip counts retain domain bound.
numeric('pythran_cdotc','''def main(private integer N, private integer[4][2] CX, private integer INCX, private integer[4][2] CY, private integer INCY) -> integer[3]:
    integer[3] result = [0; 3]
    integer ix = if INCX < 0 then (-N+1)*INCX else 0 fi
    integer iy = if INCY < 0 then (-N+1)*INCY else 0 fi
    for u32 I in 0..1024 do
        integer x = if INCX == 1 && INCY == 1 then it(I) else ix fi
        integer y = if INCX == 1 && INCY == 1 then it(I) else iy fi
        x = if x < 0 then x+4 else x fi
        y = if y < 0 then y+4 else y fi
        result[1] = if it(I) < N then result[1]+mul(CX[ui(x)][0],CY[ui(y)][0])+mul(CX[ui(x)][1],CY[ui(y)][1]) else result[1] fi
        result[2] = if it(I) < N then result[2]+mul(CX[ui(x)][0],CY[ui(y)][1])-mul(CX[ui(x)][1],CY[ui(y)][0]) else result[2] fi
        ix = ix+INCX
        iy = iy+INCY
    endfor
    result[0] = if N > 0 then 1 else 0 fi
    return result
''','kernel.py:15,24: private trip count N; @requires 0<=N<=1024 supplies a bound, including zero strides. Complex multiply needs Q32 rescaling.',rewrite=['Guard 1024 iterations with I<N.','Encode optional None/complex return with a public presence tag.'])
# Generate literal scalar loop expansion of the lattice formulas, not a new algorithm.
source=(ROOT/ENTRIES['pythran_d2q9_nxnyns']['kernel']).read_text()
tree=ast.parse(source)
def expr(e):
 if isinstance(e,ast.Constant):return str(round(e.value*Q)) if isinstance(e.value,float) else str(e.value*Q)
 if isinstance(e,ast.Name):return e.id
 if isinstance(e,ast.Subscript):return expr(e.value)+'['+str(ast.literal_eval(e.slice))+']'
 if isinstance(e,ast.UnaryOp):return '(-'+expr(e.operand)+')'
 if isinstance(e,ast.BinOp):
  if isinstance(e.op,ast.Mult):return 'mul('+expr(e.left)+','+expr(e.right)+')'
  if isinstance(e.op,ast.Div) and isinstance(e.left,ast.Constant) and isinstance(e.right,ast.Constant):return str(round(e.left.value/e.right.value*Q))
  return '('+expr(e.left)+('+' if isinstance(e.op,ast.Add) else '-')+expr(e.right)+')'
 raise ValueError(ast.dump(e))
helpers=''
for fname,args,ret in [('f2m_loc','integer[9] f, integer[9] m','m'),('relaxation_loc','integer[9] m','m'),('m2f_loc','integer[9] m, integer[9] f','f')]:
 fun=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==fname)
 helpers+=f'def {fname}({args}) -> integer[9]:\n'
 for a in fun.body:
  if isinstance(a,ast.Assign):
   lhs=expr(a.targets[0]);helpers+='    '+('integer ' if isinstance(a.targets[0],ast.Name) else '')+lhs+' = '+expr(a.value)+'\n'
  elif isinstance(a,ast.AugAssign):helpers+='    '+expr(a.target)+' = '+expr(a.target)+'+'+expr(a.value)+'\n'
 helpers+='    return '+ret+'\n\n'
body='''def main(private integer[16][16][9] f1, private integer[16][16][9] f2) -> integer[16][16][9]:
    integer[9] floc = [0; 9]
    integer[9] mloc = [0; 9]
    for u32 j in 0..16 do
        for u32 k in 0..9 do
            f1[0][j][k] = f1[14][j][k]
            f1[15][j][k] = f1[1][j][k]
        endfor
    endfor
    for u32 i in 0..16 do
        for u32 k in 0..9 do
            f1[i][0][k] = f1[i][14][k]
            f1[i][15][k] = f1[i][1][k]
        endfor
    endfor
    for u32 i in 1..15 do
        for u32 j in 1..15 do
            floc = [f1[i][j][0],f1[i-1][j][1],f1[i][j-1][2],f1[i+1][j][3],f1[i][j+1][4],f1[i-1][j-1][5],f1[i+1][j-1][6],f1[i+1][j+1][7],f1[i-1][j+1][8]]
            mloc = f2m_loc(floc,mloc)
            mloc = relaxation_loc(mloc)
            floc = m2f_loc(mloc,floc)
            for u32 k in 0..9 do
                f2[j][i][k] = floc[k]
            endfor
        endfor
    endfor
    return f2
'''
numeric('pythran_d2q9_nxnyns',helpers+body,'Source periodic boundaries, directional gathers, f2m/relaxation/m2f formulas and transposed setf are retained; Q32 native integer arithmetic.')
# Matrix complex products with real/imaginary pair encoding.
numeric('npbench_scattering_self_energies_scattering_self_energies','''def main(private integer[6][2] neigh_idx, private integer[6][2][2][3][3][2] dH, private integer[2][4][6][3][3][2] G, private integer[2][2][6][2][2][2][2] D, private integer[2][4][6][3][3][2] Sigma) -> integer[2][4][6][3][3][2]:
    for u32 k in 0..2 do
        for u32 E in 0..4 do
            for u32 q in 0..2 do
                for u32 w in 0..2 do
                    for u32 i in 0..2 do
                        for u32 j in 0..2 do
                            for u32 a in 0..6 do
                                for u32 b in 0..2 do
                                    integer nb = if neigh_idx[a][b] < 0 then neigh_idx[a][b]+6 else neigh_idx[a][b] fi
                                    u32 ew = if E >= w then E-w else 0 fi
                                    integer[3][3][2] dHG = [[[0; 2]; 3]; 3]
                                    integer[3][3][2] dHD = [[[0; 2]; 3]; 3]
                                    for u32 r in 0..3 do
                                        for u32 s in 0..3 do
                                            for u32 t in 0..3 do
                                                dHG[r][s][0] = dHG[r][s][0]+mul(G[k][ew][ui(nb)][r][t][0],dH[a][b][i][t][s][0])-mul(G[k][ew][ui(nb)][r][t][1],dH[a][b][i][t][s][1])
                                                dHG[r][s][1] = dHG[r][s][1]+mul(G[k][ew][ui(nb)][r][t][0],dH[a][b][i][t][s][1])+mul(G[k][ew][ui(nb)][r][t][1],dH[a][b][i][t][s][0])
                                            endfor
                                            dHD[r][s][0] = mul(dH[a][b][j][r][s][0],D[q][w][a][b][i][j][0])-mul(dH[a][b][j][r][s][1],D[q][w][a][b][i][j][1])
                                            dHD[r][s][1] = mul(dH[a][b][j][r][s][0],D[q][w][a][b][i][j][1])+mul(dH[a][b][j][r][s][1],D[q][w][a][b][i][j][0])
                                        endfor
                                    endfor
                                    for u32 r in 0..3 do
                                        for u32 s in 0..3 do
                                            integer real = 0
                                            integer imag = 0
                                            for u32 t in 0..3 do
                                                real = real+mul(dHG[r][t][0],dHD[t][s][0])-mul(dHG[r][t][1],dHD[t][s][1])
                                                imag = imag+mul(dHG[r][t][0],dHD[t][s][1])+mul(dHG[r][t][1],dHD[t][s][0])
                                            endfor
                                            Sigma[k][E][a][r][s][0] = if E >= w then Sigma[k][E][a][r][s][0]+real else Sigma[k][E][a][r][s][0] fi
                                            Sigma[k][E][a][r][s][1] = if E >= w then Sigma[k][E][a][r][s][1]+imag else Sigma[k][E][a][r][s][1] fi
                                        endfor
                                    endfor
                                endfor
                            endfor
                        endfor
                    endfor
                endfor
            endfor
        endfor
    endfor
    return Sigma
''','All eight source loops, complex matrix products, private neighbor reads, and E>=w mask retained. Mutated Sigma is the output.')
# One centered-residue order rotation replaces three redundant field comparisons.
for _n in ['pythran_morphology']:
 SPECS[_n]['code']=SPECS[_n]['code'].replace('return if neg(x) != neg(y) then neg(x) else x < y fi',f'return x+{P//2} < y+{P//2}')

if __name__=='__main__':
 for n in ENTRIES:
  if len(sys.argv)>1 and n not in sys.argv[1:]:continue
  execute(n)
