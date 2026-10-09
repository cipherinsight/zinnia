// pythran_ccopy__CCOPY: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each range(N) is bounded by 1024 with every iteration effect gated by active.
// Inactive array accesses use zero, so untaken branches cannot reject valid inputs.
// Indexed reads use index_pos; private writes scan CY and select every position.
// Index arithmetic is exact; only active array accesses impose index bounds.
// @outputs: float[Y_CELLS]
global NX: u32 = 4; // = CX_0
global NY: u32 = 4; // = CY_0
global PARTS: u32 = 2; // complex real/imaginary components, not a shape dimension
global X_CELLS: u32 = PARTS * NX;
global Y_CELLS: u32 = PARTS * NY;
global X_INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (CX_0 - 1) // 128)
global Y_INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (CY_0 - 1) // 128)
global MAX_STEPS: u32 = 1024; // source precondition
fn main(N: Field, CX: [Field; X_CELLS], INCX: Field, CY: [Field; Y_CELLS], INCY: Field) -> pub [Field; Y_CELLS] {
    noir_hw::check_input(N);
    noir_hw::check_inputs(CX);
    noir_hw::check_input(INCX);
    noir_hw::check_inputs(CY);
    noir_hw::check_input(INCY);
    assert(noir_hw::le(0, N));
    assert(noir_hw::le(N, MAX_STEPS as Field));
    // if N <= 0: return
    let early = noir_hw::le(N, 0);
    // if INCX == 1 and INCY == 1
    let unit = (INCX == 1) & (INCY == 1);
    let mut fast_cy = CY;
    for step in 0..MAX_STEPS {
        let active = !early & unit & noir_hw::lt(step as Field, N);
        let index = if active { step as Field } else { 0 };
        // CY[I] = CX[I]
        let source = noir_hw::index_pos::<X_INDEX_BITS>(index, NX);
        let mut value = [0; PARTS];
        for part in 0..PARTS {
            // Component address derives from a checked position.
            value[part] = CX[PARTS * source + part];
        }
        let destination = noir_hw::index_pos::<Y_INDEX_BITS>(index, NY);
        for j in 0..NY {
            let write = active & (destination == j);
            for part in 0..PARTS {
                fast_cy[PARTS * j + part] = if write { value[part] } else { fast_cy[PARTS * j + part] };
            }
        }
    }
    // else: IX = 0; IY = 0
    let mut ix: Field = 0;
    let mut iy: Field = 0;
    // if INCX < 0: IX = (-N + 1) * INCX
    let negative_x = noir_hw::lt(INCX, 0);
    let start_x = (-N + 1) * INCX;
    ix = if negative_x { start_x } else { ix };
    // if INCY < 0: IY = (-N + 1) * INCY
    let negative_y = noir_hw::lt(INCY, 0);
    let start_y = (-N + 1) * INCY;
    iy = if negative_y { start_y } else { iy };
    let mut slow_cy = CY;
    for step in 0..MAX_STEPS {
        let active = !early & !unit & noir_hw::lt(step as Field, N);
        let safe_x = if active { ix } else { 0 };
        let safe_y = if active { iy } else { 0 };
        // CY[IY] = CX[IX]
        let source = noir_hw::index_pos::<X_INDEX_BITS>(safe_x, NX);
        let mut value = [0; PARTS];
        for part in 0..PARTS { value[part] = CX[PARTS * source + part]; }
        let destination = noir_hw::index_pos::<Y_INDEX_BITS>(safe_y, NY);
        for j in 0..NY {
            let write = active & (destination == j);
            for part in 0..PARTS {
                slow_cy[PARTS * j + part] = if write { value[part] } else { slow_cy[PARTS * j + part] };
            }
        }
        // IX = IX + INCX; IY = IY + INCY
        let next_x = ix + INCX;
        ix = if active { next_x } else { ix };
        let next_y = iy + INCY;
        iy = if active { next_y } else { iy };
    }
    let mut result = CY;
    for i in 0..Y_CELLS {
        let branch = if unit { fast_cy[i] } else { slow_cy[i] };
        result[i] = if early { CY[i] } else { branch };
    }
    result
}
