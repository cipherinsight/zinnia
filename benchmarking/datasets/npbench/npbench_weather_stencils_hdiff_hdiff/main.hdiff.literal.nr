// npbench_weather_stencils_hdiff_hdiff__hdiff: literal translation of kernel.py (LITERAL_BASELINES.md).
// Every array operator is a separate full pass, including unused lap_field corners (L2).
// np.where computes its rounded product, comparison and select in separate passes.
// Only four unread input corners and overwritten out_field are omitted from input checks.
// @outputs: float[I * J * K]
global I: u32 = 11; // = out_field_0
global J: u32 = 11; // = out_field_1
global K: u32 = 10; // = out_field_2
global IN_I: u32 = I + 4;
global IN_J: u32 = J + 4;
fn main(
    in_field: [[[Field; K]; IN_J]; IN_I],
    out_field: [[[Field; K]; J]; I],
    coeff: [[[Field; K]; J]; I],
) -> pub [[[Field; K]; J]; I] {
    let _ = out_field;
    for i in 0..IN_I {
        for j in 0..IN_J {
            if !(((i == 0) | (i == IN_I - 1)) & ((j == 0) | (j == IN_J - 1))) {
                noir_hw::check_inputs(in_field[i][j]);
            }
        }
    }
    for i in 0..I {
        for j in 0..J {
            noir_hw::check_inputs(coeff[i][j]);
        }
    }
    let mut four = [[[0; K]; J + 2]; I + 2];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                four[i][j][k] = noir_hw::mul(17179869184, in_field[i + 1][j + 1][k]);
            }
        }
    }
    let mut s1 = [[[0; K]; J + 2]; I + 2];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                s1[i][j][k] = (in_field[i + 2][j + 1][k]) + (in_field[i][j + 1][k]);
            }
        }
    }
    let mut s2 = [[[0; K]; J + 2]; I + 2];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                s2[i][j][k] = (s1[i][j][k]) + (in_field[i + 1][j + 2][k]);
            }
        }
    }
    let mut s3 = [[[0; K]; J + 2]; I + 2];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                s3[i][j][k] = (s2[i][j][k]) + (in_field[i + 1][j][k]);
            }
        }
    }
    let mut lap = [[[0; K]; J + 2]; I + 2];
    for i in 0..I + 2 {
        for j in 0..J + 2 {
            for k in 0..K {
                lap[i][j][k] = (four[i][j][k]) - (s3[i][j][k]);
            }
        }
    }
    let mut resx = [[[0; K]; J]; I + 1];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                resx[i][j][k] = (lap[i + 1][j + 1][k]) - (lap[i][j + 1][k]);
            }
        }
    }
    let mut diffx = [[[0; K]; J]; I + 1];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                diffx[i][j][k] = (in_field[i + 2][j + 2][k]) - (in_field[i + 1][j + 2][k]);
            }
        }
    }
    let mut prodx = [[[0; K]; J]; I + 1];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                prodx[i][j][k] = noir_hw::mul(resx[i][j][k], diffx[i][j][k]);
            }
        }
    }
    let mut condx = [[[false; K]; J]; I + 1];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                condx[i][j][k] = noir_hw::gt(prodx[i][j][k], 0);
            }
        }
    }
    let mut flx = [[[0; K]; J]; I + 1];
    for i in 0..I + 1 {
        for j in 0..J {
            for k in 0..K {
                flx[i][j][k] = if condx[i][j][k] { 0 } else { resx[i][j][k] };
            }
        }
    }
    let mut resy = [[[0; K]; J + 1]; I];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                resy[i][j][k] = (lap[i + 1][j + 1][k]) - (lap[i + 1][j][k]);
            }
        }
    }
    let mut diffy = [[[0; K]; J + 1]; I];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                diffy[i][j][k] = (in_field[i + 2][j + 2][k]) - (in_field[i + 2][j + 1][k]);
            }
        }
    }
    let mut prody = [[[0; K]; J + 1]; I];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                prody[i][j][k] = noir_hw::mul(resy[i][j][k], diffy[i][j][k]);
            }
        }
    }
    let mut condy = [[[false; K]; J + 1]; I];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                condy[i][j][k] = noir_hw::gt(prody[i][j][k], 0);
            }
        }
    }
    let mut fly = [[[0; K]; J + 1]; I];
    for i in 0..I {
        for j in 0..J + 1 {
            for k in 0..K {
                fly[i][j][k] = if condy[i][j][k] { 0 } else { resy[i][j][k] };
            }
        }
    }
    let mut gx = [[[0; K]; J]; I];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                gx[i][j][k] = (flx[i + 1][j][k]) - (flx[i][j][k]);
            }
        }
    }
    let mut gx_y1 = [[[0; K]; J]; I];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                gx_y1[i][j][k] = (gx[i][j][k]) + (fly[i][j + 1][k]);
            }
        }
    }
    let mut divergence = [[[0; K]; J]; I];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                divergence[i][j][k] = (gx_y1[i][j][k]) - (fly[i][j][k]);
            }
        }
    }
    let mut scaled = [[[0; K]; J]; I];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                scaled[i][j][k] = noir_hw::mul(coeff[i][j][k], divergence[i][j][k]);
            }
        }
    }
    let mut result = [[[0; K]; J]; I];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                result[i][j][k] = (in_field[i + 2][j + 2][k]) - (scaled[i][j][k]);
            }
        }
    }
    result
}
