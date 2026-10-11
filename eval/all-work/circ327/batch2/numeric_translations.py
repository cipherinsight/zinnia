from translations import *
# Native Z# integers give the faithful Q32 arithmetic first attempt. No custom bit gadgets.
INTHELP='''from "EMBED" import int_to_bits, u32_from_bits, field_to_integer, u32_to_field

def ui(integer x) -> u32:
    return u32_from_bits(int_to_bits::<32>(x))

def it(u32 x) -> integer:
    return field_to_integer(u32_to_field(x))

def mul(integer x, integer y) -> integer:
    return (x*y)/4294967296

'''
def numeric(n,body,reason,rewrite=None):
 spec(n,INTHELP+body,rewrite=rewrite,reason=reason,helpers=False)
 SPECS[n]['gadget']='fixed-point multiplication/rescaling'
numeric('case0346','''def main(private integer[8][6] a, private integer row, private integer multiply_number) -> integer[6]:
    integer r = row-1
    r = if r < 0 then r+8 else r fi
    for u32 j in 0..6 do
        a[ui(r)][j] = mul(a[ui(r)][j], multiply_number)
    endfor
    integer[6] result = [0; 6]
    integer total = 0
    for u32 j in 0..6 do
        total = total+a[ui(r)][j]
        result[j] = total
    endfor
    return result
''','Private row mutation and cumsum preserved; native integer multiplication and Q32 rescale attempted.')
numeric('case0439','''def main(private integer[10][20] X) -> integer[20][10][10]:
    integer[20][10][10] result = [[[0; 10]; 10]; 20]
    for u32 k in 0..20 do
        for u32 i in 0..10 do
            for u32 j in 0..10 do
                result[k][i][j] = mul(X[i][k], X[j][k])
            endfor
        endfor
    endfor
    return result
''','Broadcast outer products; all 20x10x10 outputs retained with Q32 integer arithmetic.')
numeric('npbench_deep_learning_conv2d_bias_conv2d','''def main(private integer[4][16][16][2] input, private integer[2][2][2][8] weights, private integer[8] bias) -> integer[4][15][15][8]:
    integer[4][15][15][8] output = [[[[0; 8]; 15]; 15]; 4]
    for u32 i in 0..15 do
        for u32 j in 0..15 do
            for u32 n in 0..4 do
                for u32 c in 0..8 do
                    integer total = 0
                    for u32 di in 0..2 do
                        for u32 dj in 0..2 do
                            for u32 k in 0..2 do
                                total = total+mul(input[n][i+di][j+dj][k],weights[di][dj][k][c])
                            endfor
                        endfor
                    endfor
                    output[n][i][j][c] = total
                endfor
            endfor
        endfor
    endfor
    for u32 n in 0..4 do
        for u32 i in 0..15 do
            for u32 j in 0..15 do
                for u32 c in 0..8 do
                    output[n][i][j][c] = output[n][i][j][c]+bias[c]
                endfor
            endfor
        endfor
    endfor
    return output
''','Static convolution then bias, using native integer Q32 multiply/rescale. Source float32 storage represented in the brief-mandated fixed point.')
numeric('npbench_polybench_gesummv_gesummv','''def main(private integer alpha, private integer beta, private integer[16][16] A, private integer[16][16] B, private integer[16] x) -> integer[16]:
    integer[16] result = [0; 16]
    for u32 i in 0..16 do
        integer left = 0
        integer right = 0
        for u32 j in 0..16 do
            left = left+mul(mul(alpha,A[i][j]),x[j])
            right = right+mul(mul(beta,B[i][j]),x[j])
        endfor
        result[i] = left+right
    endfor
    return result
''','Preserve Python left associativity: (alpha*A)@x + (beta*B)@x.')
numeric('npbench_polybench_k3mm_k3mm','''def main(private integer[13][15] A, private integer[15][14] B, private integer[14][16] C, private integer[16][16] D) -> integer[13][16]:
    integer[13][14] E = [[0; 14]; 13]
    for u32 i in 0..13 do
        for u32 j in 0..14 do
            for u32 k in 0..15 do
                E[i][j] = E[i][j]+mul(A[i][k],B[k][j])
            endfor
        endfor
    endfor
    integer[13][16] F = [[0; 16]; 13]
    for u32 i in 0..13 do
        for u32 j in 0..16 do
            for u32 k in 0..14 do
                F[i][j] = F[i][j]+mul(E[i][k],C[k][j])
            endfor
        endfor
    endfor
    integer[13][16] result = [[0; 16]; 13]
    for u32 i in 0..13 do
        for u32 j in 0..16 do
            for u32 k in 0..16 do
                result[i][j] = result[i][j]+mul(F[i][k],D[k][j])
            endfor
        endfor
    endfor
    return result
''','Three matrix products in source order; Q32 integer rescaling at multiplication.')
numeric('npbench_polybench_symm_symm','''def main(private integer alpha, private integer beta, private integer[10][13] C, private integer[10][10] A, private integer[10][13] B) -> integer[10][13]:
    integer[13] temp2 = [0; 13]
    for u32 i in 0..10 do
        for u32 j in 0..13 do
            C[i][j] = mul(C[i][j],beta)
        endfor
    endfor
    for u32 i in 0..10 do
        for u32 j in 0..13 do
            for u32 k in 0..i do
                C[k][j] = C[k][j]+mul(mul(alpha,B[i][j]),A[i][k])
            endfor
            temp2[j] = 0
            for u32 k in 0..i do
                temp2[j] = temp2[j]+mul(B[k][j],A[i][k])
            endfor
        endfor
        for u32 j in 0..13 do
            C[i][j] = C[i][j]+mul(mul(alpha,B[i][j]),A[i][i])+mul(alpha,temp2[j])
        endfor
    endfor
    return C
''','Expose mutated C as public output (source returns None); keep statically unrolled triangular slices.')
numeric('pythran_allpairs','''def main(private integer[16][16] X, private integer[16][16] Y) -> integer[16][16]:
    integer[16][16] result = [[0; 16]; 16]
    for u32 j in 0..16 do
        for u32 i in 0..16 do
            for u32 k in 0..16 do
                integer diff = X[i][k]-Y[j][k]
                result[j][i] = result[j][i]+mul(diff,diff)
            endfor
        endfor
    endfor
    return result
''','Squared distances, outer Y then inner X order, with Q32 rescale.')
numeric('pythran_spectraldns','''def main(private integer[3][2][5][7][2] c, private integer[3][2][5][7] a, private integer[3][2][5][7][2] b) -> integer[3][2][5][7][2]:
    for u32 i in 0..2 do
        for u32 j in 0..5 do
            for u32 k in 0..7 do
                for u32 part in 0..2 do
                    c[0][i][j][k][part] = mul(a[0][i][j][k],b[2][i][j][k][part])-mul(a[2][i][j][k],b[1][i][j][k][part])
                    c[1][i][j][k][part] = mul(a[2][i][j][k],b[0][i][j][k][part])-mul(a[0][i][j][k],b[2][i][j][k][part])
                    c[2][i][j][k][part] = mul(a[0][i][j][k],b[1][i][j][k][part])-mul(a[1][i][j][k],b[0][i][j][k][part])
                endfor
            endfor
        endfor
    endfor
    for u32 h in 0..3 do
        for u32 i in 0..2 do
            for u32 j in 0..5 do
                for u32 k in 0..7 do
                    integer r = c[h][i][j][k][0]
                    c[h][i][j][k][0] = -c[h][i][j][k][1]
                    c[h][i][j][k][1] = r
                endfor
            endfor
        endfor
    endfor
    return c
''','Complex values encoded as real/imaginary pairs; preserve unusual source first cross component exactly.')
# Private NR/NQ/NP reshape is not expressible as written. Valid reshapes and assignment
# force NP=16 and NR*NQ=64; padded flattened intermediate has the original 1024 elements.
numeric('npbench_polybench_doitgen_doitgen','''def main(private integer NR, private integer NQ, private integer NP, private integer[8][8][16] A, private integer[16][16] C4) -> integer[8][8][16]:
    integer[8][8][16] output = [[[0; 16]; 8]; 8]
    for u32 i in 0..8 do
        for u32 j in 0..8 do
            for u32 k in 0..16 do
                for u32 l in 0..16 do
                    output[i][j][k] = output[i][j][k]+mul(A[i][j][l],C4[l][k])
                endfor
            endfor
        endfor
    endfor
    return output
''','kernel.py:10: reshape dimensions are private. The fixed C4 forces NP=16 and valid assignment forces 8x8x16; output exposes mutated A.',rewrite=['Keep the 1024-element storage and its 64 length-16 rows; runtime reshape dimensions cannot change valid row grouping.'])
