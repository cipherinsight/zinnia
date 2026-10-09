// pythran_goodexpomeasure__goodExpoMeasure: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize each source array; round-to-even uses shared divmod without local hints; uint8 cast keeps optimized backend domains.
// @outputs: int[H * W]

use noir_hw::{check_input, check_inputs, divmod_pos, canonical_exp, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global C: u32 = 16; // = inRGB_0
global H: u32 = 16; // = inRGB_1
global W: u32 = 16; // = inRGB_2

// The planes the kernel reads (R, G, B), and the centre value, from its text.
global PLANES: u32 = 3;
global HALF: Field = 2147483648; // 2^31, one half in Q32

fn main(inRGB: [[[Field; W]; H]; C], sigma: Field) -> pub [[Field; W]; H] {
    assert(C >= PLANES, "IndexError");
    // Input domain: the planes read hold int64s, sigma a Q32 code.
    for c in 0..PLANES {
        for i in 0..H {
            check_inputs(inRGB[c][i]);
        }
    }
    check_input(sigma);
    let mut r = [0; H*W]; for p in 0..H*W { r[p] = inRGB[0][p/W][p%W]*SCALE; }
    let mut g = [0; H*W]; for p in 0..H*W { g[p] = inRGB[1][p/W][p%W]*SCALE; }
    let mut b = [0; H*W]; for p in 0..H*W { b[p] = inRGB[2][p/W][p%W]*SCALE; }
    // goodExpoR = exp(-(R-128)**2/sigma)
    let mut delta = [0; H*W]; for p in 0..H*W { delta[p] = r[p]-128*SCALE; }
    let mut square = [0; H*W]; for p in 0..H*W { square[p] = mul(delta[p],delta[p]); }
    let mut negative = [0; H*W]; for p in 0..H*W { negative[p] = -square[p]; }
    let mut quotient = [0; H*W]; for p in 0..H*W { quotient[p] = noir_hw::canonical_div_nonzero(negative[p],sigma); }
    let mut er = [0; H*W]; for p in 0..H*W { er[p] = canonical_exp(quotient[p]); }
    // goodExpoG = exp(-(G-128)**2/sigma)
    let mut delta = [0; H*W]; for p in 0..H*W { delta[p] = g[p]-128*SCALE; }
    let mut square = [0; H*W]; for p in 0..H*W { square[p] = mul(delta[p],delta[p]); }
    let mut negative = [0; H*W]; for p in 0..H*W { negative[p] = -square[p]; }
    let mut quotient = [0; H*W]; for p in 0..H*W { quotient[p] = noir_hw::canonical_div_nonzero(negative[p],sigma); }
    let mut eg = [0; H*W]; for p in 0..H*W { eg[p] = canonical_exp(quotient[p]); }
    // goodExpoB = exp(-(B-128)**2/sigma)
    let mut delta = [0; H*W]; for p in 0..H*W { delta[p] = b[p]-128*SCALE; }
    let mut square = [0; H*W]; for p in 0..H*W { square[p] = mul(delta[p],delta[p]); }
    let mut negative = [0; H*W]; for p in 0..H*W { negative[p] = -square[p]; }
    let mut quotient = [0; H*W]; for p in 0..H*W { quotient[p] = noir_hw::canonical_div_nonzero(negative[p],sigma); }
    let mut eb = [0; H*W]; for p in 0..H*W { eb[p] = canonical_exp(quotient[p]); }
    let mut rg = [0; H*W]; for p in 0..H*W { rg[p]=mul(er[p],eg[p]); }
    let mut good = [0; H*W]; for p in 0..H*W { good[p]=mul(rg[p],eb[p]); }
    let mut rounded = [0; H*W]; for p in 0..H*W { rounded[p]=round_hundredths(good[p]); }
    let mut scaled = [0; H*W]; for p in 0..H*W { scaled[p]=255*rounded[p]; }
    let mut result = [[0; W]; H];
    for p in 0..H*W {
        scaled[p].assert_max_bit_size::<64>();
        let integer=divmod_pos(scaled[p],SCALE).0;
        result[p/W][p%W]=divmod_pos(integer,256).1;
    }
    result
}
fn round_hundredths(x: Field) -> Field {
    let (m,f)=divmod_pos(100*x+HALF,SCALE);
    let odd=divmod_pos(m,2).1;
    let n=if f == 0 { m-odd } else { m };
    divmod_pos(n*SCALE,100).0
}
